import pytest

from tasks.visual_media.music_transcription.document_coverage import sounding_part_inventory, validate_part_coverage


def test_inventory_uses_all_sounding_parts_and_last_sounding_measure():
    score = b'''<score-partwise><part-list>
    <score-part id="upper"><part-name>Violin</part-name><part-abbreviation>Vln</part-abbreviation></score-part>
    <score-part id="lower"><part-name>Cello</part-name></score-part>
    <score-part id="silent"><part-name>Resting percussion</part-name></score-part></part-list>
    <part id="upper"><measure number="1"><note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration></note></measure>
    <measure number="2"><note><rest/><duration>1</duration></note></measure></part>
    <part id="lower"><measure number="1"><note><rest/><duration>1</duration></note></measure>
    <measure number="2"><note><pitch><step>C</step><octave>2</octave></pitch><duration>1</duration></note></measure></part>
    <part id="silent"><measure number="1"><note><rest/><duration>2</duration></note></measure></part></score-partwise>'''
    parts = sounding_part_inventory(score)
    assert [part["part_id"] for part in parts] == ["upper", "lower"]
    assert parts[0]["last"]["measure"] == "1"
    assert parts[1]["last"] == {"measure": "2", "notes": [{"pitch": "C2", "quarters": "1", "staff": "1"}]}


def test_condensed_parts_can_share_a_page_but_each_requires_visible_evidence():
    inventory = [{"part_id": "flute1"}, {"part_id": "flute2"}]
    coverage = [{"part_id": part["part_id"], "complete": True, "first_page": 1, "last_page": 3, "staff": "Flutes 1/2, opposing voices", "evidence": "Both ending voices visible"} for part in inventory]
    assert validate_part_coverage({"part_coverage": coverage}, inventory, 3)
    coverage[1]["complete"] = False
    assert not validate_part_coverage({"part_coverage": coverage}, inventory, 3)
    coverage.pop()
    with pytest.raises(RuntimeError, match="every sounding part"):
        validate_part_coverage({"part_coverage": coverage}, inventory, 3)
