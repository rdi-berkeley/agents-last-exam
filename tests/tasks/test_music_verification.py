import io
import json
from types import SimpleNamespace

import fitz
import mido
import pytest
from PIL import Image

from tasks.visual_media.music_transcription import main, verification


class MemorySession:
    def __init__(self, files):
        self.files = files

    async def read_bytes(self, path):
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    async def file_exists(self, path):
        return path in self.files


@pytest.fixture
def delivery(monkeypatch):
    notation = b'''<score-partwise version="4.0">
      <part-list><score-part id="piano"><part-name>Piano</part-name>
      <midi-instrument id="sound"><midi-channel>1</midi-channel><midi-program>1</midi-program></midi-instrument>
      </score-part></part-list><part id="piano"><measure number="1" implicit="yes">
      <attributes><divisions>4</divisions></attributes><direction><sound tempo="120"/></direction>
      <direction><direction-type><dynamics><p/></dynamics></direction-type></direction>
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration></note>
      <direction><direction-type><dynamics><f/></dynamics></direction-type></direction>
      <note><pitch><step>D</step><octave>4</octave></pitch><duration>4</duration></note>
      </measure></part></score-partwise>'''
    sequence = mido.MidiFile(ticks_per_beat=480)
    sequence.tracks.append(mido.MidiTrack([
        mido.Message("program_change", program=0),
        mido.Message("note_on", note=60, velocity=45),
        mido.Message("note_off", note=60, time=480),
        mido.Message("note_on", note=62, velocity=100),
        mido.Message("note_off", note=62, time=480),
    ]))
    midi = io.BytesIO()
    sequence.save(file=midi)
    with fitz.open() as document:
        for label in ("First section", "Last section"):
            document.new_page().insert_text((72, 72), label)
        pdf = document.tobytes()
    screenshot = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(screenshot, "PNG")
    files = {
        "/input/task_brief.json": json.dumps({"title": "Study", "composer": "Composer", "instruments": [{"name": "Piano", "gm_program": 0}]}).encode(),
        "/reference/reference.musicxml": notation,
        "/reference/notation_reference.json": b'{"version":1,"variant":"control","groups":[]}',
        "/output/transcription.musicxml": notation,
        "/output/transcription.mid": midi.getvalue(),
        "/output/transcription.pdf": pdf,
        "/output/overview.png": screenshot.getvalue(),
    }
    task = SimpleNamespace(metadata={
        "variant_name": "control", "task_brief_path": "/input/task_brief.json",
        "reference_notation_path": "/reference/reference.musicxml",
        "reference_grouping_path": "/reference/notation_reference.json",
        "remote_output_dir": "/output",
    })
    calls = []

    async def render(*args):
        return pdf

    async def judge(**kwargs):
        calls.append(kwargs)
        return {"notation_ui": True, "same_score": True, "complete_pdf": True, "layout_score": 1, "reason": "Controlled judge response"}

    monkeypatch.setattr(verification, "render_score", render)
    monkeypatch.setattr(verification, "judge_documents", judge)
    monkeypatch.setattr(verification.EvaluationContext, "finalize", lambda self, **kwargs: None)
    return task, MemorySession(files), calls


@pytest.mark.asyncio
async def test_actual_task_entrypoint_scores_consistent_exports_and_reviews_all_pages(delivery):
    task, session, calls = delivery
    assert await main.evaluate(task, session) == pytest.approx([1])
    assert len(calls) == 1
    assert len(calls[0]["image_bytes_list"]) == 9
    assert "EVERY page" in calls[0]["prompt"]


@pytest.mark.asyncio
@pytest.mark.parametrize("filename", verification.OUTPUTS)
async def test_missing_deliverable_is_zero_without_judge_call(delivery, filename):
    task, session, calls = delivery
    del session.files[f"/output/{filename}"]
    assert await main.evaluate(task, session) == [0]
    assert not calls


@pytest.mark.asyncio
async def test_missing_reference_is_an_evaluator_failure_not_a_zero(delivery):
    task, session, _ = delivery
    del session.files["/reference/reference.musicxml"]
    with pytest.raises(FileNotFoundError):
        await main.evaluate(task, session)


@pytest.mark.asyncio
async def test_mismatched_reference_manifest_fails_explicitly(delivery):
    task, session, _ = delivery
    session.files["/reference/notation_reference.json"] = b'{"version":1,"variant":"wrong","groups":[]}'
    with pytest.raises(RuntimeError, match="manifest"):
        await main.evaluate(task, session)


@pytest.mark.asyncio
async def test_judge_authentication_failure_is_not_a_solver_zero(delivery, monkeypatch):
    task, session, _ = delivery

    async def unavailable(**kwargs):
        raise RuntimeError("Judge authentication failed")

    monkeypatch.setattr(verification, "judge_documents", unavailable)
    with pytest.raises(RuntimeError, match="authentication"):
        await main.evaluate(task, session)


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", [("notation_ui", "true"), ("same_score", None), ("layout_score", float("nan"))])
async def test_invalid_judge_response_is_not_a_valid_grade(delivery, monkeypatch, field, value):
    task, session, _ = delivery

    async def invalid(**kwargs):
        return {"notation_ui": True, "same_score": True, "complete_pdf": True, "layout_score": 1, field: value}

    monkeypatch.setattr(verification, "judge_documents", invalid)
    with pytest.raises(RuntimeError, match="Document judge"):
        await main.evaluate(task, session)


@pytest.mark.asyncio
@pytest.mark.parametrize("filename,payload", [("transcription.musicxml", b"<broken"), ("transcription.pdf", b"not a pdf"), ("overview.png", b"not an image")])
async def test_invalid_candidate_document_is_a_legitimate_zero(delivery, filename, payload):
    task, session, calls = delivery
    session.files[f"/output/{filename}"] = payload
    assert await main.evaluate(task, session) == [0]
    assert not calls


@pytest.mark.asyncio
async def test_wrong_musicxml_cannot_borrow_correct_pdf_and_midi_credit(delivery):
    task, session, _ = delivery
    session.files["/output/transcription.musicxml"] = session.files["/output/transcription.musicxml"].replace(b"<octave>4", b"<octave>5")
    assert await main.evaluate(task, session) == pytest.approx([.1])


@pytest.mark.asyncio
@pytest.mark.parametrize("suffix", ["mscz", "mscx"])
async def test_native_score_replaces_xml_without_changing_music_credit(delivery, monkeypatch, suffix):
    task, session, calls = delivery
    notation = session.files.pop("/output/transcription.musicxml")
    native_path = f"/output/transcription.{suffix}"
    session.files[native_path] = b"native controlled fixture"

    async def export(received_session, score_path, output_dir):
        assert received_session is session and score_path == native_path and output_dir == "/output"
        return notation, session.files["/output/transcription.pdf"], [(0, 120)]

    async def unexpected_render(*args):
        raise AssertionError("Native original PDF already rendered")

    monkeypatch.setattr(verification, "export_native_score", export)
    monkeypatch.setattr(verification, "render_score", unexpected_render)
    assert await main.evaluate(task, session) == pytest.approx([1])
    assert len(calls) == 1
    session.files["/output/transcription.musicxml"] = notation
    notation = notation.replace(b"<octave>4", b"<octave>5")
    assert await main.evaluate(task, session) == pytest.approx([.1])


@pytest.mark.asyncio
async def test_native_exporter_failure_is_not_a_solver_zero(delivery, monkeypatch):
    task, session, _ = delivery
    session.files["/output/transcription.mscx"] = b"native controlled fixture"

    async def unavailable(*args):
        raise RuntimeError("MuseScore exporter unavailable")

    monkeypatch.setattr(verification, "export_native_score", unavailable)
    with pytest.raises(RuntimeError, match="unavailable"):
        await main.evaluate(task, session)


@pytest.mark.asyncio
async def test_invalid_native_score_does_not_borrow_correct_xml(delivery):
    task, session, calls = delivery
    session.files["/output/transcription.mscx"] = b"<museScore><Score/></museScore>"
    assert await main.evaluate(task, session) == [0]
    assert not calls


@pytest.mark.asyncio
async def test_truncated_or_unrelated_pdf_fails_the_document_gate(delivery, monkeypatch):
    task, session, _ = delivery

    async def incomplete(**kwargs):
        return {"notation_ui": True, "same_score": True, "complete_pdf": False, "layout_score": 1}

    monkeypatch.setattr(verification, "judge_documents", incomplete)
    assert await main.evaluate(task, session) == [0]


@pytest.mark.asyncio
async def test_no_written_dynamics_uses_actual_performance_without_free_credit(delivery):
    task, session, _ = delivery
    source = session.files["/reference/reference.musicxml"]
    source = source.replace(b"<dynamics><p/></dynamics>", b"<words>Andante</words>").replace(b"<dynamics><f/></dynamics>", b"<words>cantabile</words>")
    session.files["/reference/reference.musicxml"] = source
    session.files["/output/transcription.musicxml"] = source
    session.files["/reference/reference.mid"] = session.files["/output/transcription.mid"]
    task.metadata["reference_midi_path"] = "/reference/reference.mid"
    assert await main.evaluate(task, session) == pytest.approx([1])
    sequence = mido.MidiFile(file=io.BytesIO(session.files["/output/transcription.mid"]))
    for track in sequence.tracks:
        for event in track:
            if event.type == "note_on":
                event.velocity = 64
    modified = io.BytesIO()
    sequence.save(file=modified)
    session.files["/output/transcription.mid"] = modified.getvalue()
    assert await main.evaluate(task, session) == pytest.approx([.8])
    del session.files["/reference/reference.mid"]
    with pytest.raises(FileNotFoundError):
        await main.evaluate(task, session)


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_reason", ["stop", "length"])
async def test_document_judge_requires_complete_high_effort_response(monkeypatch, finish_reason):
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish_reason, message=SimpleNamespace(content='{"same_score":true}'))])

    class Client:
        chat = SimpleNamespace(completions=SimpleNamespace(create=create))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

    monkeypatch.setattr(verification, "AsyncOpenAI", lambda **kwargs: Client())
    monkeypatch.setattr(verification, "_resolve_client_kwargs", lambda: {})
    if finish_reason == "stop":
        assert await verification.judge_documents(prompt="Review", image_bytes_list=[b"image"]) == {"same_score": True}
    else:
        with pytest.raises(RuntimeError, match="Incomplete"):
            await verification.judge_documents(prompt="Review", image_bytes_list=[b"image"])
    assert calls[0]["reasoning_effort"] == "high"
    assert calls[0]["messages"][0]["content"][1]["image_url"]["detail"] == "high"
    assert "temperature" not in calls[0]


@pytest.fixture
def ornament_delivery(delivery):
    task, session, calls = delivery
    notation = session.files["/reference/reference.musicxml"].replace(b'<divisions>4</divisions>', b'<divisions>5</divisions>').replace(b'tempo="120"', b'tempo="60"')
    notation = notation.replace(b'<duration>4</duration>', b'<duration>15</duration><notations><ornaments><trill-mark/></ornaments></notations>', 1)
    notation = notation.replace(b'<duration>4</duration>', b'<duration>5</duration>')
    session.files["/reference/reference.musicxml"] = notation
    session.files["/output/transcription.musicxml"] = notation
    sequence = mido.MidiFile(ticks_per_beat=1000)
    events = mido.MidiTrack([mido.MetaMessage("set_tempo", tempo=1000000), mido.Message("program_change", program=0)])
    previous_tick = 0
    for pitch, start, velocity in ((60, 0, 45), (62, 600, 45), (60, 1200, 45), (62, 1800, 45), (60, 2400, 45), (62, 3000, 100)):
        events.append(mido.Message("note_on", note=pitch, velocity=velocity, time=start - previous_tick))
        events.append(mido.Message("note_off", note=pitch, time=50))
        previous_tick = start + 50
    sequence.tracks = [events]
    buffer = io.BytesIO()
    sequence.save(file=buffer)
    session.files["/reference/reference.mid"] = buffer.getvalue()
    session.files["/output/transcription.mid"] = buffer.getvalue()
    task.metadata["reference_midi_path"] = "/reference/reference.mid"
    return task, session, calls


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit", [False, True])
async def test_task_entrypoint_uses_reference_trill_timing_with_written_dynamics(ornament_delivery, explicit):
    from xml.etree import ElementTree

    task, session, _ = ornament_delivery
    if explicit:
        document = ElementTree.fromstring(session.files["/output/transcription.musicxml"])
        measure = document.find("part/measure")
        original = measure.find("note")
        index = list(measure).index(original)
        measure.remove(original)
        for offset, step in enumerate(("C", "D", "C", "D", "C")):
            note = ElementTree.fromstring(ElementTree.tostring(original))
            note.remove(note.find("notations"))
            note.find("pitch/step").text = step
            note.find("duration").text = "3"
            measure.insert(index + offset, note)
        session.files["/output/transcription.musicxml"] = ElementTree.tostring(document)
    assert await main.evaluate(task, session) == pytest.approx([1])
    sequence = mido.MidiFile(file=io.BytesIO(session.files["/output/transcription.mid"]))
    elapsed = 0
    retained = []
    for event in sequence.tracks[0]:
        elapsed += event.time
        if event.type in {"note_on", "note_off"} and 600 <= elapsed <= 1250:
            continue
        retained.append((elapsed, event))
    previous = 0
    sequence.tracks[0] = mido.MidiTrack()
    for instant, event in retained:
        sequence.tracks[0].append(event.copy(time=instant - previous))
        previous = instant
    buffer = io.BytesIO()
    sequence.save(file=buffer)
    session.files["/output/transcription.mid"] = buffer.getvalue()
    assert (await main.evaluate(task, session))[0] < 1


@pytest.mark.asyncio
async def test_missing_trill_reference_is_an_evaluator_error_even_with_written_dynamics(ornament_delivery):
    task, session, calls = ornament_delivery
    del session.files["/reference/reference.mid"]
    with pytest.raises(FileNotFoundError):
        await main.evaluate(task, session)
    assert not calls


@pytest.mark.asyncio
async def test_task_entrypoint_uses_tremolo_reference_with_written_dynamics(ornament_delivery):
    task, session, _ = ornament_delivery
    for path in ("/reference/reference.musicxml", "/output/transcription.musicxml"):
        session.files[path] = session.files[path].replace(b"<trill-mark/>", b'<tremolo type="single">1</tremolo>')
    sequence = mido.MidiFile(file=io.BytesIO(session.files["/reference/reference.mid"]))
    elapsed = 0
    for event in sequence.tracks[0]:
        elapsed += event.time
        if event.type in {"note_on", "note_off"} and elapsed < 3000:
            event.note = 60
    buffer = io.BytesIO()
    sequence.save(file=buffer)
    session.files["/reference/reference.mid"] = buffer.getvalue()
    session.files["/output/transcription.mid"] = buffer.getvalue()
    assert await main.evaluate(task, session) == pytest.approx([1])
    del session.files["/reference/reference.mid"]
    with pytest.raises(FileNotFoundError):
        await main.evaluate(task, session)
