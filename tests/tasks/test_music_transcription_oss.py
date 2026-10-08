import io
import json
from pathlib import Path

import pretty_midi
import mido
import pytest

from tasks.visual_media.music_transcription import main


def midi_bytes(programs=(73, 70), pitches=((60, 62, 64), (48, 50, 52))):
    performance = pretty_midi.PrettyMIDI(initial_tempo=120)
    for program, sequence in zip(programs, pitches, strict=True):
        instrument = pretty_midi.Instrument(program)
        instrument.notes = [
            pretty_midi.Note(velocity=60 + index * 12, pitch=pitch, start=index * 0.5, end=index * 0.5 + 0.25)
            for index, pitch in enumerate(sequence)
        ]
        performance.instruments.append(instrument)
    stream = io.BytesIO()
    performance.write(stream)
    return stream.getvalue()


def test_native_linux_contract_preserves_variants_and_outputs():
    tasks = main.load()
    assert len(tasks) == 6
    for task in tasks:
        assert task.computer["setup_config"]["os_type"] == "linux"
        assert "musescore3" in task.description
        for field in ("task_brief_path", "reference_midi_path", "reference_song_mp3_path", "reference_notation_path", "reference_grouping_path"):
            assert task.metadata[field].startswith("/")
            assert "\\" not in task.metadata[field]
        assert "/transcription.pdf" in task.description
        assert "/transcription.mid" in task.description
        assert "/transcription.musicxml" in task.description
        assert "/overview.png" in task.description
    card = json.loads(Path(main.__file__).with_name("task_card.json").read_text())
    assert card["vm"]["osType"] == "linux"
    assert card["vm"]["snapshot"] == "cpu-free-ubuntu"
    assert card["requiredSystemPackages"] == ["musescore3"]
    assert card["evaluation"] == tasks[0].description[tasks[0].description.index("Evaluation weights"):].strip()


def test_reference_and_reordered_tracks_receive_identical_credit():
    reference = midi_bytes()
    reordered = midi_bytes(programs=(70, 73), pitches=((48, 50, 52), (60, 62, 64)))
    instruments = [{"gm_program": 73}, {"gm_program": 70}]
    for candidate in (reference, reordered):
        scores = main.compare_midi_unified(candidate, reference, instruments, 0.125)
        assert scores[:4] == pytest.approx((1, 1, 1, 1))


def test_absent_performance_cannot_receive_dynamics_credit():
    empty = pretty_midi.PrettyMIDI()
    stream = io.BytesIO()
    empty.write(stream)
    scores = main.compare_midi_unified(stream.getvalue(), midi_bytes(), [{"gm_program": 73}, {"gm_program": 70}], 0.125)
    assert scores[:4] == pytest.approx((0, 0, 0, 0))


def test_flat_candidate_does_not_match_varying_reference_dynamics():
    reference = [pretty_midi.Note(velocity=velocity, pitch=60, start=index, end=index + 0.5) for index, velocity in enumerate((40, 80, 120))]
    candidate = [pretty_midi.Note(velocity=64, pitch=60, start=index, end=index + 0.5) for index in range(3)]
    assert main.compute_dynamics_correlation(candidate, reference, [], [], 0.125) == 0


def test_equivalent_timing_across_rounding_boundary():
    reference = [pretty_midi.Note(velocity=90, pitch=60, start=0.049, end=0.299)]
    candidate = [pretty_midi.Note(velocity=90, pitch=60, start=0.051, end=0.301)]
    assert main.compute_pitch_f1(candidate, reference, 0.1) == 1
    assert main.compute_rhythm_f1(candidate, reference, 0.1) == 1


def test_extra_notes_and_missing_chord_notes_reduce_credit():
    reference = [pretty_midi.Note(velocity=90, pitch=pitch, start=0, end=0.5) for pitch in (60, 64, 67)]
    assert main.compute_rhythm_f1(reference[:1], reference, 0.1) == pytest.approx(0.5)
    assert main.compute_pitch_f1(reference * 2, reference, 0.1) == pytest.approx(2 / 3)


def test_wrong_pitch_and_duration_are_distinguished():
    reference = [pretty_midi.Note(velocity=90, pitch=60, start=0, end=0.5)]
    wrong_pitch = [pretty_midi.Note(velocity=90, pitch=61, start=0, end=0.5)]
    wrong_duration = [pretty_midi.Note(velocity=90, pitch=60, start=0, end=0.8)]
    assert main.compute_pitch_f1(wrong_pitch, reference, 0.1) == 0
    assert main.compute_rhythm_f1(wrong_pitch, reference, 0.1) == 1
    assert main.compute_pitch_f1(wrong_duration, reference, 0.1) == 1
    assert main.compute_rhythm_f1(wrong_duration, reference, 0.1) == 0


@pytest.mark.parametrize("port_marker_first", [False, True])
def test_multiport_programs_controls_and_tempos_remain_separate(port_marker_first):
    performance = mido.MidiFile(type=1, ticks_per_beat=480)
    performance.tracks.append(mido.MidiTrack([
        mido.MetaMessage("set_tempo", tempo=500000),
        mido.MetaMessage("set_tempo", tempo=1000000, time=480),
    ]))
    for port, program, volume in ((0, 73, 100), (1, 58, 60)):
        setup = [
            mido.Message("program_change", channel=0, program=program),
            mido.Message("control_change", channel=0, control=7, value=volume),
        ]
        marker = mido.MetaMessage("midi_port", port=port)
        setup.insert(0 if port_marker_first else len(setup), marker)
        performance.tracks.append(mido.MidiTrack([
            mido.MetaMessage("track_name", name=f"part-{port}"),
            *setup,
            mido.Message("note_on", channel=0, note=60 + port, velocity=90, time=480),
            mido.Message("note_off", channel=0, note=60 + port, velocity=0, time=480),
        ]))
    output = io.BytesIO()
    performance.save(file=output)
    tracks = main._extract_tracks_full(output.getvalue())
    assert {track.name: track.program for track in tracks} == {"part-0": 73, "part-1": 58}
    for track in tracks:
        assert track.notes[0].start == pytest.approx(0.5)
        assert track.notes[0].end == pytest.approx(1.5)
        assert track.control_changes[0].value == (100 if track.name == "part-0" else 60)


def test_midi_port_change_at_nonzero_time_keeps_absolute_note_timing():
    performance = mido.MidiFile(type=1, ticks_per_beat=480)
    performance.tracks.append(mido.MidiTrack([
        mido.Message("program_change", program=73),
        mido.Message("note_on", note=60, velocity=90),
        mido.Message("note_off", note=60, time=480),
        mido.MetaMessage("midi_port", port=1, time=480),
        mido.Message("program_change", program=58),
        mido.Message("note_on", note=62, velocity=90),
        mido.Message("note_off", note=62, time=480),
    ]))
    output = io.BytesIO()
    performance.save(file=output)
    tracks = main._extract_tracks_full(output.getvalue())
    assert [(track.program, track.notes[0].start, track.notes[0].end) for track in tracks] == [
        (73, 0.0, 0.5), (58, 1.0, 1.5),
    ]


def test_reference_track_order_does_not_change_named_instrument_expectations():
    performance = midi_bytes()
    instruments = [{"name": "Track_70", "gm_program": 70}, {"name": "Track_73", "gm_program": 73}]
    scores = main.compare_midi_unified(performance, performance, instruments, 0.125)
    assert scores[:4] == pytest.approx((1, 1, 1, 1))


def test_split_staves_share_channel_instrument_and_controller_state():
    performance = mido.MidiFile(type=1, ticks_per_beat=480)
    performance.tracks = [
        mido.MidiTrack([
            mido.MetaMessage("track_name", name="upper"),
            mido.Message("program_change", channel=3, program=46),
            mido.Message("control_change", channel=3, control=11, value=50),
            mido.Message("note_on", channel=3, note=72, velocity=90),
            mido.Message("note_off", channel=3, note=72, time=480),
            mido.Message("control_change", channel=3, control=11, value=100),
        ]),
        mido.MidiTrack([
            mido.MetaMessage("track_name", name="lower"),
            mido.Message("note_on", channel=3, note=48, velocity=90),
            mido.Message("note_off", channel=3, note=48, time=960),
        ]),
    ]
    output = io.BytesIO()
    performance.save(file=output)
    tracks = main._extract_tracks_full(output.getvalue())
    assert {track.name: track.program for track in tracks} == {"upper": 46, "lower": 46}
    for track in tracks:
        assert [(event.number, event.value, event.time) for event in track.control_changes] == [(11, 50, 0.0), (11, 100, 0.5)]
