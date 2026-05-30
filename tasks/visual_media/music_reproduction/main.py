"""Music Reproduction — 1 recovered variant.

Reproduce the accompaniment/backing track of a recorded piece as a DAW
project from audio alone.  The agent listens to a reference audio,
identifies instrumental parts (ignoring vocals and non-reproducible elements),
enters MIDI, assigns virtual instruments, and exports MIDI, audio mixdown,
per-track stems, and a screenshot.

Evaluation is stem-driven: only the k instruments that have reference stems
are scored (where k = number of reference stem files).  The reference MIDI
may contain more tracks than available stems; unmatched tracks are ignored.

Two-layer approach — MIDI layer for composition accuracy (pitch/rhythm F1,
instrument assignment) and audio layer for dynamics (RMS loudness contour
correlation) and timbre (MFCC cosine similarity).

Recovered variant: piano_with_organ.
"""

import io
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
TASK_NAME = "music_reproduction"

REFERENCE_AUDIO = "reference_audio.wav"
REFERENCE_MIDI_FILE = "reference.mid"
REFERENCE_STEMS_DIR = "stems"

TRANSCRIPTION_MIDI = "transcription.mid"
MIXDOWN_WAV = "mixdown.wav"
STEMS_DIR = "stems"
OVERVIEW_SCREENSHOT = "overview.png"

KEYSWITCH_PITCH_THRESHOLD = 24  # C1 — notes below this are keyswitches

# GM programs 0–15: Piano (0–7) + Chromatic Percussion (8–15).
# These instruments legitimately use low pitches (e.g., piano C0/E0) so
# keyswitch filtering is skipped. They also commonly use CC64 (sustain pedal),
# so note durations are normalized to effective sounding duration.
KEYBOARD_PROGRAMS = set(range(16))

# Scoring weights
W_PITCH = 0.25
W_RHYTHM = 0.25
W_DYNAMICS = 0.10
W_INSTRUMENT = 0.10
W_TIMBRE = 0.20
W_ARRANGEMENT = 0.10

# ---------------------------------------------------------------------------
# Variants — (task_tag,)
# Each variant has its own input/ and reference/ in GCS at:
#   gs://ale-data-all/visual_media/music_reproduction/<task_tag>/input/
#   gs://ale-data-all/visual_media/music_reproduction/<task_tag>/reference/
# ---------------------------------------------------------------------------
VARIANTS = [
    ("piano_with_organ",),
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


def _is_keyboard_program(program: int | None) -> bool:
    """Check if a GM program belongs to the keyboard family (0–15)."""
    return program is not None and program in KEYBOARD_PROGRAMS


def _filter_keyswitches(
    notes: list[pretty_midi.Note],
) -> list[pretty_midi.Note]:
    """Remove keyswitch notes (pitch < C1 / MIDI 24)."""
    return [n for n in notes if n.pitch >= KEYSWITCH_PITCH_THRESHOLD]


def _normalize_cc64_durations(
    notes: list[pretty_midi.Note],
    control_changes: list[pretty_midi.ControlChange],
    track_end_time: float = float("inf"),
) -> list[pretty_midi.Note]:
    """Resolve sustain pedal (CC64) into effective note durations.

    When CC64 is active (value >= 64) at a note's note-off time, the note's
    effective duration extends until CC64 releases or the next note-on of the
    same pitch, whichever comes first.
    """
    cc64_events = sorted(
        [(cc.time, cc.value) for cc in control_changes if cc.number == 64],
        key=lambda x: x[0],
    )
    if not cc64_events:
        return notes

    def _cc64_active_at(t: float) -> bool:
        """Check if CC64 is active (>= 64) at time t."""
        val = 0
        for ct, cv in cc64_events:
            if ct > t:
                break
            val = cv
        return val >= 64

    def _next_cc64_off_after(t: float) -> float:
        """Find the next time CC64 goes below 64 after time t."""
        for ct, cv in cc64_events:
            if ct > t and cv < 64:
                return ct
        return float("inf")

    # Build a map of next note-on per pitch for same-pitch collision detection
    sorted_notes = sorted(notes, key=lambda n: n.start)
    pitch_onsets: dict[int, list[float]] = {}
    for n in sorted_notes:
        pitch_onsets.setdefault(n.pitch, []).append(n.start)

    result = []
    for n in sorted_notes:
        if not _cc64_active_at(n.end):
            result.append(n)
            continue

        # CC64 is active at note-off — extend duration
        pedal_off = _next_cc64_off_after(n.end)

        # Find next note-on of the same pitch
        onsets = pitch_onsets.get(n.pitch, [])
        next_same_pitch = float("inf")
        for onset in onsets:
            if onset > n.start:
                next_same_pitch = onset
                break

        effective_end = min(pedal_off, next_same_pitch, track_end_time)
        new_note = pretty_midi.Note(
            velocity=n.velocity,
            pitch=n.pitch,
            start=n.start,
            end=effective_end if effective_end != float("inf") else n.end,
        )
        result.append(new_note)

    return result


def _extract_tracks_full(midi_bytes: bytes, preprocess: bool = True) -> list[TrackInfo]:
    """Extract all tracks with full metadata from MIDI bytes.

    When preprocess=True:
    - Keyboard instruments (GM 0–15): CC64 duration normalization, NO keyswitch filter
    - Other instruments: keyswitch filter (pitch < 24), NO CC64 normalization
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
    except Exception as e:
        logger.warning(f"Failed to parse MIDI: {e}")
        return []

    track_end_time = midi_data.get_end_time()

    tracks: list[TrackInfo] = []
    for inst in midi_data.instruments:
        name = inst.name.strip() if inst.name else f"Track_{inst.program}"
        if inst.is_drum:
            name = f"Drums_{name}"
        program = None if inst.is_drum else int(inst.program)
        notes = list(inst.notes)
        ccs = list(inst.control_changes)

        if preprocess:
            if _is_keyboard_program(program):
                # Keyboard: normalize CC64, keep all pitches
                notes = _normalize_cc64_durations(notes, ccs, track_end_time)
            else:
                # Non-keyboard: filter keyswitches, no CC64 normalization
                notes = _filter_keyswitches(notes)

        tracks.append(
            TrackInfo(
                name=name,
                program=program,
                is_drum=inst.is_drum,
                notes=notes,
                control_changes=ccs,
            )
        )
    return tracks


def _get_min_duration(midi_bytes: bytes) -> float:
    """Find the shortest note duration in the MIDI file for quantization.

    Skips keyswitch notes for non-keyboard instruments only.
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
    except Exception:
        raise ValueError("Failed to parse reference MIDI bytes.")

    min_dur = float("inf")
    for inst in midi_data.instruments:
        program = None if inst.is_drum else int(inst.program)
        is_keyboard = _is_keyboard_program(program)
        for note in inst.notes:
            if not is_keyboard and note.pitch < KEYSWITCH_PITCH_THRESHOLD:
                continue
            dur = note.end - note.start
            if dur > 0.01 and dur < min_dur:
                min_dur = dur

    if min_dur == float("inf"):
        raise ValueError("No valid notes found in reference MIDI.")

    return min_dur


def _get_tempo_bpm(midi_bytes: bytes) -> float:
    """Extract tempo BPM from reference MIDI. Defaults to 120 if not found."""
    try:
        midi_data = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
        tempo_changes = midi_data.get_tempo_changes()
        if len(tempo_changes[1]) > 0:
            return float(tempo_changes[1][0])
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


AUDIO_EXTENSIONS = (".wav", ".mp3")
REMOTE_SCORE_SCRIPT = Path(__file__).parent / "scripts" / "score_audio_remote.py"


def _find_stem_file(stem_files: list[str], track_name: str) -> str | None:
    """Fuzzy-match a MIDI track name to a stem filename."""
    name_lower = track_name.lower().strip()
    for f in stem_files:
        base = os.path.splitext(f)[0].lower().strip()
        if base == name_lower:
            return f
    for f in stem_files:
        base = os.path.splitext(f)[0].lower().strip()
        if name_lower in base or base in name_lower:
            return f
    return None


def compare_midi_only(
    agent_midi_bytes: bytes,
    ref_midi_bytes: bytes,
    resolution: float,
    agent_stem_files: list[str],
    ref_stem_files: list[str],
) -> tuple[float, float, float, list[dict]]:
    """MIDI-layer evaluation: pitch F1, rhythm F1, instrument assignment.

    Also produces stem pairings for the remote audio scoring step.
    Returns (avg_pitch_f1, avg_rhythm_f1, instrument_score, per_track_details).
    """
    agent_tracks = _extract_tracks_full(agent_midi_bytes, preprocess=True)
    ref_tracks = _extract_tracks_full(ref_midi_bytes, preprocess=True)

    if not ref_tracks:
        logger.warning("Reference MIDI has no tracks")
        return 0.0, 0.0, 0.0, []

    matched = _match_tracks_hungarian(agent_tracks, ref_tracks, resolution)

    details: list[dict] = []
    pitch_scores: list[float] = []
    rhythm_scores: list[float] = []
    instrument_correct = 0
    instrument_total = 0

    for ref_track, agent_track, match_pitch_f1 in matched:
        ref_stem_name = _find_stem_file(ref_stem_files, ref_track.name)
        if ref_stem_name is None:
            details.append(
                {
                    "ref_track": ref_track.name,
                    "agent_track": agent_track.name if agent_track else "(unmatched)",
                    "skipped": True,
                    "reason": "no reference stem",
                }
            )
            continue

        agent_notes = agent_track.notes if agent_track else []
        ref_notes = ref_track.notes

        pitch_f1 = match_pitch_f1
        rhythm_f1 = compute_rhythm_f1(agent_notes, ref_notes, resolution)
        pitch_scores.append(pitch_f1)
        rhythm_scores.append(rhythm_f1)

        ref_program = ref_track.program
        agent_program = agent_track.program if agent_track else None
        is_program_correct = False
        if ref_program is not None:
            instrument_total += 1
            is_program_correct = agent_program is not None and agent_program == ref_program
            if is_program_correct:
                instrument_correct += 1

        agent_stem_name = None
        if agent_track:
            agent_stem_name = _find_stem_file(agent_stem_files, agent_track.name)

        details.append(
            {
                "ref_track": ref_track.name,
                "agent_track": agent_track.name if agent_track else "(unmatched)",
                "skipped": False,
                "pitch_f1": round(pitch_f1, 4),
                "rhythm_f1": round(rhythm_f1, 4),
                "agent_notes": len(agent_notes),
                "ref_notes": len(ref_notes),
                "expected_program": ref_program,
                "actual_program": agent_program,
                "program_correct": is_program_correct,
                "agent_stem": agent_stem_name,
                "ref_stem": ref_stem_name,
            }
        )

    avg_pitch = sum(pitch_scores) / len(pitch_scores) if pitch_scores else 0.0
    avg_rhythm = sum(rhythm_scores) / len(rhythm_scores) if rhythm_scores else 0.0
    instrument_score = instrument_correct / instrument_total if instrument_total > 0 else 0.0

    return avg_pitch, avg_rhythm, instrument_score, details


# ---------------------------------------------------------------------------
# Task config
# ---------------------------------------------------------------------------
@dataclass
class TaskConfig(GeneralTaskConfig):
    DOMAIN_NAME: str = "visual_media"

    TASK_NAME: str = "music_reproduction"
    VARIANT_NAME: str = ""  # Set per variant

    @property
    def task_dir(self) -> str:
        """Use the canonical benchmark runtime root on Windows data disk."""
        return rf"E:\agenthle\visual_media\music_reproduction\{self.VARIANT_NAME}"

    @property
    def input_dir(self) -> str:
        return rf"{self.task_dir}\input"

    @property
    def reference_audio_path(self) -> str:
        return rf"{self.input_dir}\{REFERENCE_AUDIO}"

    @property
    def reference_midi_path(self) -> str:
        return rf"{self.reference_dir}\{REFERENCE_MIDI_FILE}"

    @property
    def reference_stems_dir(self) -> str:
        return rf"{self.reference_dir}\{REFERENCE_STEMS_DIR}"

    @property
    def task_description(self) -> str:
        return f"""\
Goal: Reproduce the accompaniment/backing track of a piece of music as a DAW project \
from audio alone. Listen to the reference audio, identify all instrumental parts \
(ignore vocals and other non-reproducible elements), tempo, key, and time signature, \
then recreate the instrumental accompaniment using any DAW available on the system \
(e.g., Cubase, Logic Pro, FL Studio, Ableton Live) with appropriate virtual instruments.

Input:
- {self.reference_audio_path} — the reference audio recording (stereo). \
This is the ONLY input. No metadata or task brief is provided. \
You must infer tempo, key, time signature, and instrumentation from the audio. \
Focus on reproducing the instrumental/accompaniment parts only — skip vocals.

Steps:
1. Listen to the reference audio to identify all instrumental parts, tempo, key, and time signature. \
Ignore vocals and any non-reproducible elements.
2. Open any DAW available on the system and create a new empty project.
3. Set the project tempo and time signature based on what you hear.
4. Create one MIDI/Instrument track per identified instrument, naming each track after the instrument.
5. Assign an appropriate virtual instrument to each track. \
The patch does not need to be identical to the original, but must belong to the correct instrument family.
6. Transcribe all notes, rhythms, and dynamics for each instrument by entering MIDI data.
7. Adjust velocity and expression to reflect the dynamic contour of each part.
8. Export outputs:
   a. Export MIDI file > save as {self.remote_output_dir}\\{TRANSCRIPTION_MIDI}
   b. Export audio mixdown (WAV stereo) > save as {self.remote_output_dir}\\{MIXDOWN_WAV}
   c. For EACH track: solo the track, export audio mixdown > \
save as {self.remote_output_dir}\\{STEMS_DIR}\\<track_name>.wav
   d. Save the DAW project file in the project's own location.
9. Take a screenshot of the DAW arrange/timeline view showing all tracks with MIDI regions visible:
   save_milestone_screenshot(path="{self.remote_output_dir}\\{OVERVIEW_SCREENSHOT}", \
description="DAW arrange view with all tracks")

Output files (all saved to {self.remote_output_dir}):
- {TRANSCRIPTION_MIDI}: Exported MIDI file with GM program assignments per track.
- {MIXDOWN_WAV}: Stereo audio mixdown.
- {STEMS_DIR}/: Per-track solo WAV exports (one file per instrument track).
- {OVERVIEW_SCREENSHOT}: Screenshot of the DAW arrange/timeline view.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "input_dir": self.input_dir,
                "reference_audio_path": self.reference_audio_path,
                "reference_midi_path": self.reference_midi_path,
                "reference_stems_dir": self.reference_stems_dir,
            }
        )
        return metadata


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------
@cb.tasks_config(split="train")
def load():
    """Register the recovered music reproduction variants."""
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
    """Score: Gates (MIDI + mixdown + stems + screenshot),
    then pitch 0.25 + rhythm 0.25 + dynamics 0.10 + instrument 0.10
    + timbre 0.20 + arrangement 0.10.

    MIDI analysis runs locally (small files). WAV analysis (dynamics, timbre,
    mixdown checks) runs on the remote VM via score_audio_remote.py.
    """
    try:
        meta = task_cfg.metadata
        tag = meta["variant_name"]
        output_dir = meta["remote_output_dir"]
        ref_dir = meta["reference_dir"]
        reference_midi_path = meta["reference_midi_path"]
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
            # Gate 1: transcription.mid exists, >= 1 note
            # -----------------------------------------------------------
            if TRANSCRIPTION_MIDI not in output_files:
                logger.warning(f"{TRANSCRIPTION_MIDI} not found — gate fail")
                ctx.log_evaluation(identifier="gate_midi", score=0.0, error="MIDI not found")
                ctx.finalize(num_output_files=len(output_files))
                return [0.0]

            agent_midi_path = os.path.join(output_dir, TRANSCRIPTION_MIDI)
            agent_midi_bytes = await session.read_bytes(agent_midi_path)
            agent_tracks_check = _extract_tracks_full(agent_midi_bytes, preprocess=True)
            total_notes = sum(len(t.notes) for t in agent_tracks_check)
            if total_notes == 0:
                logger.warning("MIDI has 0 notes after keyswitch filtering — gate fail")
                ctx.log_evaluation(
                    identifier="gate_midi",
                    score=0.0,
                    error="0 notes after keyswitch filtering",
                )
                ctx.finalize(num_output_files=len(output_files))
                return [0.0]

            ctx.log_evaluation(identifier="gate_midi", score=1.0)

            # -----------------------------------------------------------
            # Gate 2: mixdown.wav exists
            # -----------------------------------------------------------
            if MIXDOWN_WAV not in output_files:
                logger.warning(f"{MIXDOWN_WAV} not found — gate fail")
                ctx.log_evaluation(identifier="gate_mixdown", score=0.0, error="Mixdown not found")
                ctx.finalize(num_output_files=len(output_files))
                return [0.0]

            mixdown_path = os.path.join(output_dir, MIXDOWN_WAV)

            # -----------------------------------------------------------
            # Gate 3: stems/ directory exists with WAV files
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
            # Gate 4: screenshot shows DAW arrange/timeline view
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

            prompt_screenshot = (
                "You are evaluating a DAW (Digital Audio Workstation) screenshot.\n\n"
                "Does this image show a DAW (Digital Audio Workstation) arrange or "
                "timeline view with MIDI instrument tracks visible, showing MIDI "
                "regions/clips on the timeline?\n"
                "(Look for: track list with instrument names, MIDI clips/regions on "
                "a timeline, transport bar, mixer or inspector panels)\n"
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
                logger.warning("Screenshot gate failed — does not show DAW arrange view")
                ctx.finalize(num_output_files=len(output_files))
                return [0.0]

            # -----------------------------------------------------------
            # Load reference MIDI (small file, stays local)
            # -----------------------------------------------------------
            ref_midi_bytes = None
            if await session.exists(reference_midi_path):
                ref_midi_bytes = await session.read_bytes(reference_midi_path)

            if not ref_midi_bytes:
                logger.error("Reference MIDI not found — cannot score")
                ctx.finalize(num_output_files=len(output_files))
                return [0.0]

            # List reference stems (filenames only — no download)
            ref_stem_files: list[str] = []
            if await session.exists(reference_stems_dir):
                ref_dir_list = await session.list_dir(reference_stems_dir)
                ref_stem_files = [f for f in ref_dir_list if f.lower().endswith(AUDIO_EXTENSIONS)]

            # -----------------------------------------------------------
            # MIDI-layer scoring (local)
            # -----------------------------------------------------------
            bpm = _get_tempo_bpm(ref_midi_bytes)
            min_note_sec = _get_min_duration(ref_midi_bytes)
            sixteenth_sec = (60.0 / bpm) * 0.25
            resolution_sec = min(min_note_sec, sixteenth_sec)

            avg_pitch_f1, avg_rhythm_f1, inst_score, track_details = compare_midi_only(
                agent_midi_bytes,
                ref_midi_bytes,
                resolution_sec,
                wav_stems,
                ref_stem_files,
            )

            # Build pairings for remote audio scoring
            pairings = []
            for d in track_details:
                if d.get("skipped"):
                    continue
                if d.get("agent_stem") and d.get("ref_stem"):
                    pairings.append({"ref_stem": d["ref_stem"], "agent_stem": d["agent_stem"]})

            # -----------------------------------------------------------
            # Remote audio scoring: upload script, run on VM, pull JSON
            # -----------------------------------------------------------
            remote_script = rf"{output_dir}\__score_audio_remote.py"
            remote_result = rf"{output_dir}\__eval_result.json"
            remote_pairings = rf"{output_dir}\__pairings.json"

            await session.write_bytes(remote_script, REMOTE_SCORE_SCRIPT.read_bytes())
            await session.write_bytes(remote_pairings, json.dumps(pairings).encode("utf-8"))

            cmd = (
                f'python "{remote_script}" '
                f'--agent-stems-dir "{stems_output_dir}" '
                f'--ref-stems-dir "{reference_stems_dir}" '
                f'--mixdown-path "{mixdown_path}" '
                f'--pairings-json "{remote_pairings}" '
                f'--result-path "{remote_result}"'
            )
            logger.info(f"[{tag}] Running remote audio scoring: {cmd}")
            await session.run_command(cmd, timeout=300)

            result_bytes = await session.read_bytes(remote_result)
            audio_result = json.loads(result_bytes)

            if "error" in audio_result:
                logger.error(f"Remote scoring error: {audio_result['error']}")
                ctx.finalize(num_output_files=len(output_files))
                return [0.0]

            # -----------------------------------------------------------
            # Mixdown gate from remote result
            # -----------------------------------------------------------
            mixdown_info = audio_result["mixdown"]
            if mixdown_info["file_size"] < 51200:
                logger.warning("Mixdown too small — gate fail")
                ctx.log_evaluation(
                    identifier="gate_mixdown",
                    score=0.0,
                    error="Mixdown < 50 KB",
                )
                ctx.finalize(num_output_files=len(output_files))
                return [0.0]

            if mixdown_info["rms_db"] <= -60.0:
                logger.warning(
                    f"Mixdown is silent (RMS={mixdown_info['rms_db']:.1f} dB) — gate fail"
                )
                ctx.log_evaluation(
                    identifier="gate_mixdown",
                    score=0.0,
                    error=f"Mixdown silent (RMS={mixdown_info['rms_db']:.1f} dB)",
                )
                ctx.finalize(num_output_files=len(output_files))
                return [0.0]

            ctx.log_evaluation(identifier="gate_mixdown", score=1.0)

            # Stems gate from remote result
            num_non_silent = audio_result["num_non_silent"]
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
            # Merge audio scores into track details
            # -----------------------------------------------------------
            pair_scores_map = {}
            for ps in audio_result.get("pair_scores", []):
                key = (ps["ref_stem"], ps["agent_stem"])
                pair_scores_map[key] = ps

            dynamics_vals: list[float] = []
            timbre_vals: list[float] = []

            for d in track_details:
                if d.get("skipped"):
                    continue
                key = (d.get("ref_stem"), d.get("agent_stem"))
                if key in pair_scores_map:
                    ps = pair_scores_map[key]
                    d["dynamics"] = ps["dynamics"]
                    d["timbre"] = ps["timbre"]
                    dynamics_vals.append(ps["dynamics"])
                    timbre_vals.append(ps["timbre"])
                else:
                    d["dynamics"] = 0.5
                    d["timbre"] = 0.0
                    dynamics_vals.append(0.5)
                    timbre_vals.append(0.0)

            avg_dynamics = sum(dynamics_vals) / len(dynamics_vals) if dynamics_vals else 0.5
            avg_timbre = sum(timbre_vals) / len(timbre_vals) if timbre_vals else 0.0

            # -----------------------------------------------------------
            # Scoring
            # -----------------------------------------------------------
            pitch_score = W_PITCH * avg_pitch_f1
            ctx.add_score(pitch_score)
            ctx.log_evaluation(
                identifier="pitch_accuracy",
                score=pitch_score,
                pitch_f1=round(avg_pitch_f1, 4),
            )

            rhythm_score = W_RHYTHM * avg_rhythm_f1
            ctx.add_score(rhythm_score)
            ctx.log_evaluation(
                identifier="rhythm_accuracy",
                score=rhythm_score,
                rhythm_f1=round(avg_rhythm_f1, 4),
            )

            dynamics_score = W_DYNAMICS * avg_dynamics
            ctx.add_score(dynamics_score)
            ctx.log_evaluation(
                identifier="dynamics",
                score=dynamics_score,
                dynamics_corr=round(avg_dynamics, 4),
            )

            instrument_score = W_INSTRUMENT * inst_score
            ctx.add_score(instrument_score)
            ctx.log_evaluation(
                identifier="instrument_assignment",
                score=instrument_score,
                assignment_rate=round(inst_score, 4),
            )

            timbre_score = W_TIMBRE * avg_timbre
            ctx.add_score(timbre_score)
            ctx.log_evaluation(
                identifier="timbre_consistency",
                score=timbre_score,
                timbre_similarity=round(avg_timbre, 4),
            )

            logger.info(
                f"Comparison: pitch_f1={avg_pitch_f1:.4f} "
                f"rhythm_f1={avg_rhythm_f1:.4f} dynamics={avg_dynamics:.4f} "
                f"instrument={inst_score:.4f} timbre={avg_timbre:.4f}"
            )
            for d in track_details:
                if d.get("skipped"):
                    logger.info(f"  [SKIP] Ref '{d['ref_track']}' — {d.get('reason', 'no stem')}")
                    continue
                prog_status = "OK" if d.get("program_correct") else "MISS"
                logger.info(
                    f"  [{prog_status}] Ref '{d['ref_track']}' <-> Agent "
                    f"'{d['agent_track']}': pitch={d['pitch_f1']:.4f} "
                    f"rhythm={d['rhythm_f1']:.4f} dynamics={d.get('dynamics', 0):.4f} "
                    f"timbre={d.get('timbre', 0):.4f} "
                    f"(agent:{d['agent_notes']}, ref:{d['ref_notes']}) "
                    f"stems: agent={d['agent_stem']} ref={d['ref_stem']}"
                )

            # Arrangement quality (0.10) — LLM vision judge
            prompt_arrangement = (
                "You are evaluating a DAW project screenshot.\n\n"
                "Assess the arrangement quality of this project:\n"
                "- Does it have a reasonable number of tracks for the music?\n"
                "- Are tracks logically named after instruments?\n"
                "- Are tracks organized in a professional manner "
                "(e.g., grouped by instrument family)?\n"
                "- Are MIDI regions/clips visible on the timeline?\n\n"
                "Does this project show professional arrangement quality?\n"
                'Answer with ONLY "YES" or "NO".'
            )

            eval_arrangement = await llm_vision_judge(
                prompt=prompt_arrangement,
                image_bytes=screenshot_bytes,
                reference_image_bytes=None,
                return_details=True,
                max_tokens=10,
                eval_context=ctx,
                identifier="arrangement_quality",
            )
            arrangement_score = W_ARRANGEMENT * eval_arrangement["score"]
            ctx.add_score(arrangement_score)

            ctx.finalize(
                num_stems=num_non_silent,
                num_output_files=len(output_files),
                bpm=bpm,
                resolution_sec=round(resolution_sec, 6),
            )

            total = ctx.total_score
            logger.info(
                f"Final score: {total:.4f} "
                f"(pitch={pitch_score:.3f} rhythm={rhythm_score:.3f} "
                f"dynamics={dynamics_score:.3f} instrument={instrument_score:.3f} "
                f"timbre={timbre_score:.3f} arrangement={arrangement_score:.3f})"
            )
            return [total]

    except Exception as e:
        logger.error(f"Evaluation error: {e}")

    return [0.0]
