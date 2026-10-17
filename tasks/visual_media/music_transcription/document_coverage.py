from fractions import Fraction

from .notation import _read_document


def sounding_part_inventory(content):
    document = _read_document(content)
    definitions = {part.get("id"): part for part in document.findall("part-list/score-part")}
    inventory = []
    for part in document.findall("part"):
        divisions = 1
        sounding = []
        for index, measure in enumerate(part.findall("measure"), start=1):
            divisions = int(measure.findtext("attributes/divisions", str(divisions)))
            notes = []
            for note in measure.findall("note"):
                if note.find("rest") is not None or note.find("cue") is not None:
                    continue
                pitch = note.find("pitch")
                written = "unpitched" if pitch is None else "".join(pitch.findtext(field, "") for field in ("step", "alter", "octave"))
                notes.append({"pitch": written, "quarters": str(Fraction(note.findtext("duration", "0")) / divisions), "staff": note.findtext("staff", "1")})
            if notes:
                sounding.append({"measure": measure.get("number", str(index)), "notes": notes})
        if not sounding:
            continue
        definition = definitions[part.get("id")]
        inventory.append({
            "part_id": part.get("id"),
            "name": definition.findtext("part-name", part.get("id")),
            "abbreviation": definition.findtext("part-abbreviation", ""),
            "sounding_measures": [measure["measure"] for measure in sounding],
            "first": sounding[0],
            "last": sounding[-1],
        })
    return inventory


def validate_part_coverage(response, inventory, page_count):
    coverage = response.get("part_coverage")
    expected = {part["part_id"] for part in inventory}
    if not isinstance(coverage, list) or len(coverage) != len(expected):
        raise RuntimeError("Document judge did not review every sounding part")
    seen = set()
    for entry in coverage:
        if not isinstance(entry, dict) or entry.get("part_id") not in expected or entry["part_id"] in seen:
            raise RuntimeError("Document judge returned invalid part identities")
        seen.add(entry["part_id"])
        if type(entry.get("complete")) is not bool or not isinstance(entry.get("evidence"), str) or not entry["evidence"].strip():
            raise RuntimeError("Document judge returned invalid part evidence")
        if entry["complete"]:
            for field in ("first_page", "last_page"):
                if type(entry.get(field)) is not int or not 1 <= entry[field] <= page_count:
                    raise RuntimeError("Document judge omitted submitted-page coverage")
            if not isinstance(entry.get("staff"), str) or not entry["staff"].strip():
                raise RuntimeError("Document judge omitted visible staff identification")
    return all(entry["complete"] for entry in coverage)
