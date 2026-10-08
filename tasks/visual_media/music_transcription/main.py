"""Music Transcription — 6 canonical variants.

Transcribe a recorded piece into sheet music using any music notation software,
export consistent MusicXML, PDF and MIDI with correct instrument assignments.
Evaluation: notation pitch/rhythm F1, relative dynamics, instrument assignment,
MIDI consistency, and whole-score document review.

Variants: Dorico Prelude, Fugue 16, Iconica, Liebestraume, Triumphant, Unshaken.
"""

import io
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass

# Workaround: cua_bench loads main.py via exec_module without registering
# in sys.modules, which causes @dataclass to fail. Register ourselves.
if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))

import cua_bench as cb
import mido
import pretty_midi

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig
from tasks.visual_media.music_transcription.verification import evaluate_delivery

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
REFERENCE_NOTATION = "reference.musicxml"
REFERENCE_GROUPING = "notation_reference.json"

TRANSCRIPTION_PDF = "transcription.pdf"
TRANSCRIPTION_MIDI = "transcription.mid"
TRANSCRIPTION_NOTATION = "transcription.musicxml"
OVERVIEW_SCREENSHOT = "overview.png"

# ---------------------------------------------------------------------------
# Variants — (task_tag,)
# Each variant has its own input/ and reference/ staged by the framework.
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
        source = mido.MidiFile(file=io.BytesIO(midi_bytes))
        if source.type not in (0, 1) or source.ticks_per_beat <= 0:
            raise ValueError("Expected synchronous MIDI with positive tick resolution")
        tempo_events = []
        port_tracks = defaultdict(list)
        for track in source.tracks:
            initial_port = 0
            for event in track:
                if event.time:
                    break
                if event.type == "midi_port":
                    initial_port = event.port
                    break
            port = initial_port
            tick = 0
            events_by_port = defaultdict(list)
            for event in track:
                tick += event.time
                if event.type == "midi_port":
                    port = event.port
                elif event.type == "set_tempo":
                    tempo_events.append((tick, event))
                elif hasattr(event, "channel") or event.type == "sysex":
                    events_by_port[port].append((tick, event))
            for port, events in events_by_port.items():
                converted = mido.MidiTrack([mido.MetaMessage("track_name", name=track.name)])
                previous_tick = 0
                for tick, event in events:
                    converted.append(event.copy(time=tick - previous_tick))
                    previous_tick = tick
                converted.append(mido.MetaMessage("end_of_track"))
                port_tracks[port].append(converted)
        performances = []
        for tracks in port_tracks.values():
            channel_state = []
            timed_tracks = []
            for track_index, track in enumerate(tracks):
                tick = 0
                notes = []
                for event_index, event in enumerate(track):
                    tick += event.time
                    if event.type in ("program_change", "control_change", "pitchwheel"):
                        channel_state.append((tick, 0, track_index, event_index, event))
                    elif event.type in ("note_on", "note_off"):
                        notes.append((tick, 1, track_index, event_index, event))
                timed_tracks.append(notes)
            shared_tracks = []
            for track, notes in zip(tracks, timed_tracks, strict=True):
                if not notes:
                    continue
                channels = {entry[-1].channel for entry in notes}
                events = notes + [entry for entry in channel_state if entry[-1].channel in channels]
                converted = mido.MidiTrack([mido.MetaMessage("track_name", name=track.name)])
                previous_tick = 0
                for tick, _, _, _, event in sorted(events, key=lambda entry: entry[:4]):
                    converted.append(event.copy(time=tick - previous_tick))
                    previous_tick = tick
                shared_tracks.append(converted)
            timeline = mido.MidiTrack()
            previous_tick = 0
            for tick, event in sorted(tempo_events, key=lambda entry: entry[0]):
                timeline.append(event.copy(time=tick - previous_tick))
                previous_tick = tick
            isolated = mido.MidiFile(type=1, ticks_per_beat=source.ticks_per_beat)
            isolated.tracks = [timeline, *shared_tracks]
            stream = io.BytesIO()
            isolated.save(file=stream)
            performances.append(pretty_midi.PrettyMIDI(io.BytesIO(stream.getvalue())))
    except Exception as e:
        logger.warning(f"Failed to parse MIDI: {e}")
        return []

    tracks: list[TrackInfo] = []
    for inst in (instrument for performance in performances for instrument in performance.instruments):
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
    tracks = _extract_tracks_full(midi_bytes)
    if not tracks:
        raise ValueError("Failed to parse reference MIDI bytes.")

    min_dur = float("inf")
    for inst in tracks:
        for note in inst.notes:
            dur = note.end - note.start
            if dur > 0.01 and dur < min_dur:
                min_dur = dur

    if min_dur == float("inf"):
        raise ValueError("No valid notes found in reference MIDI.")

    return min_dur


def _match_notes(agent_notes, ref_notes, resolution, *, rhythm=False):
    """Match notes one-to-one within the reference timing resolution."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    if not agent_notes or not ref_notes:
        return []
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("Timing resolution must be finite and positive")
    onset_error = abs(
        np.array([note.start for note in ref_notes])[:, None]
        - np.array([note.start for note in agent_notes])
    )
    if rhythm:
        duration_error = abs(
            np.array([note.end - note.start for note in ref_notes])[:, None]
            - np.array([note.end - note.start for note in agent_notes])
        )
        error = np.maximum(onset_error, duration_error)
        eligible = error <= resolution + 1e-10
    else:
        error = onset_error
        eligible = (error <= resolution + 1e-10) & (
            np.array([note.pitch for note in ref_notes])[:, None]
            == np.array([note.pitch for note in agent_notes])
        )
    proximity = 1 / (1 + error / resolution)
    weights = eligible * (1 + proximity / (min(len(ref_notes), len(agent_notes)) + 1))
    reference_indices, agent_indices = linear_sum_assignment(weights, maximize=True)
    return [
        (int(agent_index), int(reference_index))
        for reference_index, agent_index in zip(reference_indices, agent_indices, strict=True)
        if eligible[reference_index, agent_index]
    ]


def compute_pitch_f1(
    agent_notes: list[pretty_midi.Note],
    ref_notes: list[pretty_midi.Note],
    resolution: float,
) -> float:
    """Note-level F1 on pitch and onset within the timing tolerance."""
    if not ref_notes:
        return 1.0 if not agent_notes else 0.0
    if not agent_notes:
        return 0.0

    matched = _match_notes(agent_notes, ref_notes, resolution)
    return 2 * len(matched) / (len(agent_notes) + len(ref_notes))


def compute_rhythm_f1(
    agent_notes: list[pretty_midi.Note],
    ref_notes: list[pretty_midi.Note],
    resolution: float,
) -> float:
    """Note-level F1 on onset and duration within the timing tolerance."""
    if not ref_notes:
        return 1.0 if not agent_notes else 0.0
    if not agent_notes:
        return 0.0

    matched = _match_notes(agent_notes, ref_notes, resolution, rhythm=True)
    return 2 * len(matched) / (len(agent_notes) + len(ref_notes))


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

    if not agent_notes:
        return 1.0 if not ref_notes else 0.0

    scores: list[float] = []

    matched = _match_notes(agent_notes, ref_notes, resolution)
    if not matched:
        return 0.0
    agent_vels = [agent_notes[agent_index].velocity for agent_index, _ in matched]
    ref_vels = [ref_notes[reference_index].velocity for _, reference_index in matched]

    if len(agent_vels) >= 3 and len(set(ref_vels)) > 1:
        if len(set(agent_vels)) == 1:
            scores.append(0.0)
        else:
            corr, _ = spearmanr(agent_vels, ref_vels)
            scores.append(0.0 if np.isnan(corr) else max(0.0, (corr + 1.0) / 2.0))

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
                scores.append(0.0 if np.isnan(corr) else max(0.0, (corr + 1.0) / 2.0))

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
    expected_programs = {
        entry["name"]: entry.get("gm_program")
        for entry in expected_instruments
        if "name" in entry
    }

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
        exp_program = expected_programs.get(ref_track.name.removeprefix("Drums_"))
        if not expected_programs and ref_idx < len(expected_instruments):
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
    avg_dynamics = sum(dynamics_scores) / len(dynamics_scores) if dynamics_scores else 0.0
    instrument_score = instrument_correct / instrument_total if instrument_total > 0 else 0.0

    return avg_pitch, avg_rhythm, avg_dynamics, instrument_score, details


# ---------------------------------------------------------------------------
# Task config
# ---------------------------------------------------------------------------
@dataclass
class TaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = "visual_media"

    TASK_NAME: str = "music_transcription"
    VARIANT_NAME: str = ""  # Set per variant

    @property
    def input_dir(self) -> str:
        return f"{self.task_dir}/input"

    @property
    def task_brief_path(self) -> str:
        return f"{self.input_dir}/{TASK_BRIEF_FILE}"

    @property
    def reference_song_mp3_path(self) -> str:
        return f"{self.input_dir}/{REFERENCE_SONG_MP3}"

    @property
    def reference_midi_path(self) -> str:
        return f"{self.reference_dir}/{REFERENCE_MIDI_FILE}"

    @property
    def reference_score_pdf_path(self) -> str:
        return f"{self.reference_dir}/{REFERENCE_SCORE_PDF}"

    @property
    def reference_notation_path(self) -> str:
        return f"{self.reference_dir}/{REFERENCE_NOTATION}"

    @property
    def reference_grouping_path(self) -> str:
        return f"{self.reference_dir}/{REFERENCE_GROUPING}"

    @property
    def task_description(self) -> str:
        return f"""\
Goal: Transcribe a recorded piece into musical notation, export both a PDF score and a MIDI file with correct instrument assignments. You may use any music notation software available on the system.

MuseScore 3 is installed on Linux; launch it with `musescore3`. You may use the
editor or scripts. Preserve the complete orchestration, rather than a melody
reduction. The PDF, MusicXML and MIDI must represent the same complete musical work.
MusicXML is the machine-readable export of your score, not a separate transcription.
Alternatively, save the complete MuseScore project as transcription.mscz or
transcription.mscx instead of MusicXML. The evaluator exports that score itself.
The evaluator uses .mscz, then .mscx, then MusicXML, in that order.

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
   - Export a PDF of the full score. Save it as {self.remote_output_dir}/{TRANSCRIPTION_PDF}
   - Export MIDI: save as {self.remote_output_dir}/{TRANSCRIPTION_MIDI}
   - Save the native MuseScore score as {self.remote_output_dir}/transcription.mscz (or .mscx),
     or export MusicXML (plain or compressed) as {self.remote_output_dir}/{TRANSCRIPTION_NOTATION}
10. Take ONE final screenshot showing the notation software with the score visible (showing some instrument staves):
    save_milestone_screenshot(path="{self.remote_output_dir}/{OVERVIEW_SCREENSHOT}", description="Overview of the transcribed score")

Output files (all saved to {self.remote_output_dir}):
- {TRANSCRIPTION_PDF}: Exported PDF of the complete score with all parts.
- {TRANSCRIPTION_MIDI}: Exported MIDI file. Each track must have correct MIDI Program Change messages matching the GM program numbers in task_brief.json.
- transcription.mscz or transcription.mscx, or {TRANSCRIPTION_NOTATION}: The same complete editable score.
- {OVERVIEW_SCREENSHOT}: Screenshot of the notation software showing the transcribed score with instrument staves visible.

Evaluation weights are pitch 30%, rhythm 30%, dynamics 20%, instrument assignment
10%, and score layout 10%. Missing parts lose credit. Track order and names,
equivalent MIDI encodings, correctly
transposed notation, and alternative readable page layouts are unrestricted.
GM program numbers in the brief are zero-based; follow the explicit numbers.
Pitch and rhythm use one-to-one matching of sounding pitches and performed
onsets from your MusicXML, with a 40-millisecond tolerance. Extra and missing
notes reduce F1. Rhythm checks written duration, duration articulations, arpeggios,
trills, pedal spans, glissando endpoints and legato phrasing.
Instrument credit includes audible playing techniques such
as harmonics, muting and pizzicato, with equivalent standard words or symbols.
Written rhythm is checked
rather than the synthesizer's note-release length. Measured tremolos and their
explicit repeated notes are equivalent. Dynamics compares relative written
loudness and crescendo/diminuendo profiles (80%) and force articulations (20%).
For a part without written dynamic markings in the source score, the loudness
profile is assessed from matched MIDI note velocities and volume/expression
controls against the recording's reference performance. Ornament attacks are
grouped by their written note, allowing equivalent complete roll rates and
grace timing without changing the required loudness pattern. Overall volume scaling
is accepted; missing, flat or reversed expression loses credit. Missing or extra
performed notes reduce this comparison's coverage.
The four musical categories are averaged over the brief's logical instruments;
splitting an instrument across staves or files' MIDI tracks changes no weight.
MIDI coverage and precision against your own score scale the musical scores by
their harmonic mean. Wrong MIDI programs also reduce instrument credit.
For notated arpeggios, MIDI must play every chord pitch in the marked order,
with its first or last attack within 40 ms of the written onset.
Grace notes must be played in their written order, with a separate attack for
each repeated grace pitch. They may precede or delay the principal note: the
first grace or principal attack must align within 40 ms of its written onset.
Grace groups and arpeggios stay within their local score passage, bounded by
the preceding onset and the principal/chord ending or next onset in the
participating voices. Their timing has no universal duration fraction or
seconds cap. Ordinary notes retain the same 40 ms accuracy requirement.
Trills must alternate between the written note and its indicated neighbor,
either using a trill mark or writing out the complete alternating notes,
spanning the written duration. Either pitch may start the trill within 40 ms
of its onset. Playback gaps and the final attack are checked against that
passage in the reference performance, with the same 40 ms tolerance, rather
than a universal trill speed. A held note is not a performed trill.
Glissandi must play intervening pitches in the marked direction, not just their
endpoints. Complete written-out trills and monotonic glissando runs are
equivalent to their marks when their endpoints match the source span and
their written notes connect within 40 ms. Tremolo playback preserves repeated
pitches and alternating phases. Timing follows local tempo and the written
subdivisions, using the reference passage's cadence and complete span when
repeated source attacks are available. Complete alternative roll rates are
accepted; skipped attacks and incomplete spans lose credit.
Piano pedal markings apply across the instrument's staves. MIDI sustain
controllers or equivalent extended note releases must preserve those sustained
regions; pedal transitions use the same 40 ms tolerance.
The complete PDF must contain the same score as the MusicXML; different page
layouts and valid condensed staves are accepted. The screenshot must show a
notation editor with the score visible. Missing, invalid or inconsistent
required documents receive zero. A correct but incomplete transcription can
receive partial musical credit if its submitted exports agree.
Layout receives full credit for readable professional notation with correct
identification and no material collisions or clipping, half credit for usable
notation with specific readability defects, and zero if unusable or unidentified.
Cover pages, sparse orchestration and alternative spacing are not defects.
For D.C., D.S., coda and fine, submit your native MuseScore score. The evaluator
expands playback navigation; you do not need to rewrite repeated passages.
Ordinary repeats, numbered endings and transposed instruments are supported.

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
                "reference_notation_path": self.reference_notation_path,
                "reference_grouping_path": self.reference_grouping_path,
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


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    return await evaluate_delivery(task_cfg, session, _extract_tracks_full)
