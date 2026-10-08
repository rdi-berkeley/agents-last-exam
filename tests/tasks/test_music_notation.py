import io
import zipfile
from fractions import Fraction

import pytest

from tasks.visual_media.music_transcription.notation import dynamic_values, parse_musicxml


def score_bytes(measures, attributes="", instruments=""):
    return f'''<score-partwise version="4.0">
      <work><work-title>Control</work-title></work>
      <identification><creator type="composer">Composer</creator></identification>
      <part-list><score-part id="part"><part-name>Clarinet</part-name>
        <score-instrument id="clarinet"><instrument-name>Clarinet</instrument-name></score-instrument>
        <midi-instrument id="clarinet"><midi-channel>1</midi-channel><midi-program>72</midi-program></midi-instrument>
        {instruments}
      </score-part></part-list>
      <part id="part"><measure number="1" implicit="yes">
        <attributes><divisions>12</divisions>{attributes}</attributes>
        {measures}
      </measure></part></score-partwise>'''.encode()


def test_sounding_pitch_chords_ties_and_tempo():
    content = score_bytes('''
      <direction><sound tempo="90"/></direction>
      <note><pitch><step>D</step><octave>4</octave></pitch><duration>12</duration><tie type="start"/><notations><articulations><tenuto/></articulations></notations></note>
      <note><chord/><pitch><step>F</step><alter>1</alter><octave>4</octave></pitch><duration>12</duration></note>
      <note><pitch><step>D</step><octave>4</octave></pitch><duration>12</duration><tie type="stop"/></note>
    ''', '<transpose><diatonic>-1</diatonic><chromatic>-2</chromatic></transpose>')
    score = parse_musicxml(content)
    assert score.title == "Control" and score.composer == "Composer"
    notes = score.parts[0].notes
    assert [(note.pitch, note.start, note.end, note.program) for note in notes] == [(60, 0, 2, 71), (64, 0, 1, 71)]
    assert all(note.articulations == {"tenuto"} for note in notes)
    assert score.seconds(Fraction(2)) == pytest.approx(4 / 3)


def test_polyphonic_backups_and_exact_tuplet_durations():
    content = score_bytes('''
      <note><pitch><step>C</step><octave>5</octave></pitch><duration>4</duration><voice>1</voice></note>
      <note><pitch><step>D</step><octave>5</octave></pitch><duration>4</duration><voice>1</voice></note>
      <note><pitch><step>E</step><octave>5</octave></pitch><duration>4</duration><voice>1</voice></note>
      <backup><duration>12</duration></backup>
      <note><pitch><step>C</step><octave>3</octave></pitch><duration>12</duration><voice>2</voice></note>
    ''')
    notes = parse_musicxml(content).parts[0].notes
    assert [(note.pitch, note.start, note.end) for note in notes] == [
        (48, 0, 1), (72, 0, Fraction(1, 3)),
        (74, Fraction(1, 3), Fraction(2, 3)), (76, Fraction(2, 3), 1),
    ]


def test_repeat_first_and_second_endings():
    content = score_bytes('''
      <barline location="left"><repeat direction="forward"/></barline>
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>12</duration></note>
      </measure><measure number="2" implicit="yes">
      <barline location="left"><ending type="start" number="1"/></barline>
      <note><pitch><step>D</step><octave>4</octave></pitch><duration>12</duration></note>
      <barline location="right"><ending type="stop" number="1"/><repeat direction="backward"/></barline>
      </measure><measure number="3" implicit="yes">
      <barline location="left"><ending type="start" number="2"/></barline>
      <note><pitch><step>E</step><octave>4</octave></pitch><duration>12</duration></note>
      <barline location="right"><ending type="stop" number="2"/></barline>
    ''')
    notes = parse_musicxml(content).parts[0].notes
    assert [(note.pitch, note.start) for note in notes] == [(60, 0), (62, 1), (60, 2), (64, 3)]


def test_percussion_uses_instrument_identity_and_preserves_roll():
    content = score_bytes('''
      <note><unpitched><display-step>B</display-step><display-octave>4</display-octave></unpitched>
        <duration>48</duration><instrument id="drum"/>
        <notations><ornaments><tremolo type="single">3</tremolo></ornaments></notations>
      </note>
    ''', instruments='<midi-instrument id="drum"><midi-channel>10</midi-channel><midi-program>1</midi-program><midi-unpitched>36</midi-unpitched></midi-instrument>')
    note = parse_musicxml(content).parts[0].notes[0]
    assert (note.pitch, note.program, note.percussion, note.tremolo, note.end) == (35, None, True, 3, 4)


def test_dynamic_offsets_and_shared_endpoints_survive_polyphony():
    content = score_bytes('''
      <direction><direction-type><dynamics><p/></dynamics></direction-type><offset>6</offset><staff>2</staff></direction>
      <direction><direction-type><wedge type="crescendo" number="1"/></direction-type><staff>2</staff></direction>
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>12</duration></note>
      <direction><direction-type><wedge type="stop" number="1"/></direction-type><staff>2</staff></direction>
    ''')
    part = parse_musicxml(content).parts[0]
    assert [(mark.kind, mark.value, mark.position, mark.staff) for mark in part.marks] == [
        ("wedge", "crescendo", 0, "2"), ("dynamic", "p", Fraction(1, 2), "2"), ("wedge", "stop", 1, "2"),
    ]


def test_grace_notes_and_duplicates_are_not_dropped():
    content = score_bytes('''
      <note><grace/><pitch><step>D</step><octave>4</octave></pitch></note>
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>12</duration></note>
      <note><chord/><pitch><step>C</step><octave>4</octave></pitch><duration>12</duration></note>
    ''')
    notes = parse_musicxml(content).parts[0].notes
    assert len(notes) == 3
    assert sum(note.pitch == 60 for note in notes) == 2
    assert next(note for note in notes if note.pitch == 62).articulations == {"grace"}
    assert all("grace" not in note.articulations for note in notes if note.pitch == 60)


def test_grace_articulation_stays_on_its_own_chord():
    content = score_bytes('''
      <note><grace/><pitch><step>D</step><octave>4</octave></pitch><notations><articulations><staccato/></articulations></notations></note>
      <note><grace/><pitch><step>E</step><octave>4</octave></pitch></note>
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>12</duration><notations><articulations><tenuto/></articulations></notations></note>
      <note><chord/><pitch><step>G</step><octave>4</octave></pitch><duration>12</duration></note>
    ''')
    notes = {note.pitch: note for note in parse_musicxml(content).parts[0].notes}
    assert notes[62].articulations == {"grace", "staccato"}
    assert notes[64].articulations == {"grace"}
    assert notes[60].articulations == notes[67].articulations == {"tenuto"}
    assert (notes[62].grace_order, notes[64].grace_order, notes[60].grace_order) == (0, 1, None)


def test_compressed_musicxml_matches_plain_xml():
    content = score_bytes('<note><pitch><step>C</step><octave>4</octave></pitch><duration>12</duration></note>')
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("META-INF/container.xml", '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="score.musicxml"/></rootfiles></container>')
        archive.writestr("score.musicxml", content)
    assert parse_musicxml(stream.getvalue()) == parse_musicxml(content)


@pytest.mark.parametrize("fragment", [
    '<backup><duration>12</duration></backup>',
    '<note><pitch><step>C</step><octave>4</octave></pitch><duration>12</duration><tie type="stop"/></note>',
    '<note><pitch><step>C</step><octave>4</octave></pitch><duration>0</duration></note>',
])
def test_invalid_notation_fails_explicitly(fragment):
    with pytest.raises(ValueError):
        parse_musicxml(score_bytes(fragment))


def test_octave_shift_does_not_transpose_already_sounding_pitch():
    content = score_bytes('''
      <direction><direction-type><octave-shift type="down" size="8"/></direction-type></direction>
      <note><pitch><step>C</step><octave>6</octave></pitch><duration>12</duration></note>
      <direction><direction-type><octave-shift type="stop" size="8"/></direction-type></direction>
    ''')
    assert parse_musicxml(content).parts[0].notes[0].pitch == 84


def test_score_timewise_and_partwise_have_identical_music():
    from xml.etree import ElementTree

    source = score_bytes('<note><pitch><step>C</step><octave>4</octave></pitch><duration>12</duration></note>')
    document = ElementTree.fromstring(source)
    part = document.find("part")
    measure = part.find("measure")
    document.remove(part)
    document.tag = "score-timewise"
    replacement = ElementTree.SubElement(document, "measure", measure.attrib)
    replacement_part = ElementTree.SubElement(replacement, "part", part.attrib)
    replacement_part.extend(measure)
    assert parse_musicxml(ElementTree.tostring(document)) == parse_musicxml(source)


def test_dynamic_hairpins_interpolate_and_preserve_the_arrival_level():
    content = score_bytes('''
      <direction><direction-type><dynamics><p/></dynamics></direction-type></direction>
      <direction><direction-type><wedge type="crescendo" number="2"/></direction-type></direction>
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>48</duration></note>
      <direction><direction-type><wedge type="stop" number="2"/></direction-type></direction>
      <direction><direction-type><dynamics><f/></dynamics></direction-type></direction>
      <note><pitch><step>D</step><octave>4</octave></pitch><duration>12</duration></note>
    ''')
    part = parse_musicxml(content).parts[0]
    assert dynamic_values(part, "1", [0, 1, 2, 3, 4, 5]) == [3, 3.75, 4.5, 5.25, 6, 6]
    assert dynamic_values(part, "2", [0, 1, 2]) == [None, None, None]


def test_word_crescendo_and_redundant_static_marks_are_equivalent():
    music = '''
      <direction><direction-type><dynamics><p/></dynamics></direction-type></direction>
      <direction><direction-type><words>cresc.</words></direction-type></direction>
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>24</duration></note>
      <direction><direction-type><dynamics><f/></dynamics></direction-type></direction>
    '''
    part = parse_musicxml(score_bytes(music)).parts[0]
    assert dynamic_values(part, "1", [0, 1, 2]) == [3, 4.5, 6]


def test_unsupported_navigation_is_reported_instead_of_silently_omitted():
    music = '<direction><sound dacapo="yes"/></direction><note><pitch><step>C</step><octave>4</octave></pitch><duration>12</duration></note>'
    with pytest.raises(ValueError, match="navigation expanded"):
        parse_musicxml(score_bytes(music))


def test_new_measure_tempo_overrides_previous_measure_end_reset():
    music = '''
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>12</duration></note>
      <sound tempo="86.05"/>
      </measure><measure number="2" implicit="yes">
      <sound tempo="35"/>
      <note><pitch><step>D</step><octave>4</octave></pitch><duration>12</duration></note>
    '''
    score = parse_musicxml(score_bytes(music))
    assert score.tempos == [(Fraction(1), 35)]
    assert score.seconds(Fraction(2)) == pytest.approx(.5 + 60 / 35)


def test_conflicting_tempos_across_parts_are_still_rejected():
    from xml.etree import ElementTree

    document = ElementTree.fromstring(score_bytes('''
      <sound tempo="90"/>
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>12</duration></note>
    '''))
    second = ElementTree.fromstring(ElementTree.tostring(document.find('part')))
    second.find('measure/sound').set('tempo', '60')
    document.append(second)
    with pytest.raises(ValueError, match='across parts'):
        parse_musicxml(ElementTree.tostring(document))


def test_native_playback_tempo_restores_events_omitted_by_xml_export():
    content = score_bytes('''
      <direction><sound tempo="120"/></direction>
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>24</duration></note>
    ''')
    score = parse_musicxml(content, tempo_override=[(Fraction(0), 120), (Fraction(1), 60)])
    assert score.seconds(Fraction(2)) == pytest.approx(1.5)
    assert score.parts[0].notes[0].end == 2
    with pytest.raises(ValueError, match='tempo map'):
        parse_musicxml(content, tempo_override=[(Fraction(0), 0)])


@pytest.mark.parametrize("offset, sound, expected", [
    ('<offset>6</offset>', '<sound tempo="90"/>', Fraction(0)),
    ('<offset sound="yes">6</offset>', '<sound tempo="90"/>', Fraction(1, 2)),
    ('<offset sound="yes">6</offset>', '<sound tempo="90"><offset>3</offset></sound>', Fraction(1, 4)),
])
def test_visual_and_playback_tempo_offsets_follow_musicxml_semantics(offset, sound, expected):
    music = f'<direction><direction-type><words>Slow</words></direction-type>{offset}{sound}</direction>'
    music += '<note><pitch><step>C</step><octave>4</octave></pitch><duration>12</duration></note>'
    score = parse_musicxml(score_bytes(music))
    assert score.tempos == [(expected, 90)]
    assert score.parts[0].marks[0].position == Fraction(1, 2)


def test_slur_numbers_follow_document_order_across_overlapping_voices():
    from tasks.visual_media.music_transcription.notation_scoring import compare_notation

    music = '''
      <note><pitch><step>C</step><octave>5</octave></pitch><duration>12</duration><voice>1</voice><notations><slur type="start" number="1"/></notations></note>
      <note><pitch><step>D</step><octave>5</octave></pitch><duration>12</duration><voice>1</voice><notations><slur type="stop" number="1"/></notations></note>
      <backup><duration>24</duration></backup>
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>6</duration><voice>2</voice></note>
      <note><pitch><step>D</step><octave>4</octave></pitch><duration>6</duration><voice>2</voice><notations><slur type="start" number="1"/></notations></note>
      <note><pitch><step>E</step><octave>4</octave></pitch><duration>12</duration><voice>2</voice><notations><slur type="stop" number="1"/></notations></note>
    '''
    candidate = parse_musicxml(score_bytes(music))
    opening, second_voice = music.split("<backup>", 1)
    reference = parse_musicxml(score_bytes(opening + "<backup>" + second_voice.replace('number="1"', 'number="2"')))
    numbers = {mark.number for mark in candidate.parts[0].marks if mark.kind == "slur"}
    assert len(numbers) == 2
    assert compare_notation(candidate, reference)["rhythm"] == 1


def test_cross_staff_slur_can_stop_before_start_in_document_order():
    from tasks.visual_media.music_transcription.notation_scoring import expanded_notes

    music = '''
      <forward><duration>12</duration></forward>
      <note><pitch><step>C</step><octave>5</octave></pitch><duration>12</duration><voice>1</voice><staff>1</staff><notations><slur type="stop" number="1"/></notations></note>
      <backup><duration>24</duration></backup>
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>24</duration><voice>2</voice><staff>2</staff><notations><slur type="start" number="1"/></notations></note>
    '''
    score = parse_musicxml(score_bytes(music))
    assert all("legato" in entry.articulations for entry in expanded_notes(score.parts[0]))


def test_unmatched_span_is_not_accepted_as_valid_notation():
    music = '<note><pitch><step>C</step><octave>4</octave></pitch><duration>12</duration><notations><slur type="start" number="1"/></notations></note>'
    with pytest.raises(ValueError, match="Unmatched notation span"):
        parse_musicxml(score_bytes(music))


def test_arpeggio_direction_and_shared_cross_staff_identity_are_preserved():
    music = '''
      <note><pitch><step>C</step><octave>5</octave></pitch><duration>12</duration><staff>1</staff><notations><arpeggiate number="2" direction="down"/></notations></note>
      <backup><duration>12</duration></backup>
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>12</duration><staff>2</staff><notations><arpeggiate number="2" direction="down"/></notations></note>
    '''
    score = parse_musicxml(score_bytes(music))
    assert all(note.arpeggio == ("2", "down") for note in score.parts[0].notes)
    assert {note.staff for note in score.parts[0].notes} == {"1", "2"}
