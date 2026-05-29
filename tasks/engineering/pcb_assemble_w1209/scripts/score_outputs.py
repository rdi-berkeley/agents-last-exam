"""Local scorer for engineering/pcb_assemble_w1209."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_BOM = "W1209_BOM.csv"
REQUIRED_STEP = "W1209_assembled.step"
STEP_MIN_REFERENCE_SIZE_FRACTION = 0.25
STEP_MIN_ASSEMBLY_USAGE_FRACTION = 0.75
STEP_MIN_ASSEMBLY_USAGE_ABSOLUTE = 20
STEP_MIN_GEOMETRY_ENTITY_FRACTION = 0.30
STEP_MIN_MANIFOLD_SOLID_FRACTION = 0.50
OCC_MIN_FACE_FRACTION = 0.30
OCC_MIN_SOLID_FRACTION = 0.50
DETAIL_MAX_MISSING_REFERENCE_RECORDS = 20
STEP_QUOTED_FIELD_RE = re.compile(r"'((?:[^']|'')*)'")
STEP_REFERENCE_RE = re.compile(r"#(\d+)")
STEP_ENTITY_NAME_RE = re.compile(r"^\s*([A-Z0-9_]+)\s*\(", re.IGNORECASE)
GEOMETRY_ENTITY_NAMES = (
    "CARTESIAN_POINT",
    "ADVANCED_FACE",
    "MANIFOLD_SOLID_BREP",
    "PRODUCT_DEFINITION",
)


@dataclass(frozen=True)
class StepRecord:
    entity: str
    args: str
    refs: tuple[int, ...]
    quoted_fields: tuple[str, ...]


def _result(score: float, reasons: list[str], details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "score": float(score),
        "passed": score >= 1.0,
        "reasons": reasons,
        "details": details or {},
    }


def _normalize_name(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.lower().endswith(".step"):
        cleaned = cleaned[:-5]
    aliases = {
        "BUTTON_IT_1102WD": "BUTTON_IT-1102WD",
        "button_it_1102wd": "BUTTON_IT-1102WD",
        "button_it-1102wd": "BUTTON_IT-1102WD",
    }
    return aliases.get(cleaned, aliases.get(cleaned.lower(), cleaned))


def _parse_quantity(raw: str) -> int:
    value = raw.strip()
    if not value:
        raise ValueError("empty quantity")
    number = float(value)
    if not number.is_integer():
        raise ValueError(f"non-integer quantity {raw!r}")
    quantity = int(number)
    if quantity < 0:
        raise ValueError(f"negative quantity {raw!r}")
    return quantity


def _load_bom(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("missing CSV header")
        missing = [column for column in ("Item", "Name", "Quantity") if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"missing required BOM columns: {missing}")

        counts: dict[str, int] = {}
        for row_index, row in enumerate(reader, start=2):
            name = _normalize_name(row.get("Name", ""))
            if not name:
                raise ValueError(f"row {row_index} has empty Name")
            quantity = _parse_quantity(row.get("Quantity", ""))
            counts[name] = counts.get(name, 0) + quantity
    if not counts:
        raise ValueError("BOM has no item rows")
    return counts


def _parse_step_records(path: Path) -> dict[int, StepRecord]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    records: dict[int, StepRecord] = {}
    idx = 0
    length = len(text)
    while idx < length:
        hash_pos = text.find("#", idx)
        if hash_pos < 0:
            break
        cursor = hash_pos + 1
        while cursor < length and text[cursor].isdigit():
            cursor += 1
        if cursor == hash_pos + 1:
            idx = hash_pos + 1
            continue
        record_id = int(text[hash_pos + 1 : cursor])
        while cursor < length and text[cursor].isspace():
            cursor += 1
        if cursor >= length or text[cursor] != "=":
            idx = cursor
            continue
        cursor += 1
        body_start = cursor
        depth = 0
        in_string = False
        while cursor < length:
            char = text[cursor]
            if char == "'":
                if in_string and cursor + 1 < length and text[cursor + 1] == "'":
                    cursor += 2
                    continue
                in_string = not in_string
            elif not in_string:
                if char == "(":
                    depth += 1
                elif char == ")" and depth > 0:
                    depth -= 1
                elif char == ";" and depth == 0:
                    break
            cursor += 1
        if cursor >= length:
            break
        body = text[body_start:cursor].strip()
        entity_match = STEP_ENTITY_NAME_RE.match(body)
        entity = entity_match.group(1).upper() if entity_match else "COMPLEX_ENTITY"
        records[record_id] = StepRecord(
            entity=entity,
            args=body,
            refs=tuple(int(raw) for raw in STEP_REFERENCE_RE.findall(body)),
            quoted_fields=tuple(
                field.replace("''", "'") for field in STEP_QUOTED_FIELD_RE.findall(body)
            ),
        )
        idx = cursor + 1
    return records


def _count_entity(records: dict[int, StepRecord], entity: str) -> int:
    entity_upper = entity.upper()
    return sum(1 for record in records.values() if record.entity == entity_upper)


def _assembly_usage_counts(records: dict[int, StepRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records.values():
        if record.entity != "NEXT_ASSEMBLY_USAGE_OCCURRENCE":
            continue
        if len(record.quoted_fields) < 2:
            continue
        component_name = _normalize_name(record.quoted_fields[1])
        if component_name:
            counts[component_name] = counts.get(component_name, 0) + 1
    return counts


def _geometry_entity_counts(records: dict[int, StepRecord]) -> dict[str, int]:
    return {entity: _count_entity(records, entity) for entity in GEOMETRY_ENTITY_NAMES}


def _missing_references(
    records: dict[int, StepRecord],
    entities: tuple[str, ...],
) -> dict[int, list[int]]:
    entity_set = set(entities)
    record_ids = set(records)
    missing: dict[int, list[int]] = {}
    for record_id, record in records.items():
        if record.entity not in entity_set:
            continue
        absent = sorted({ref for ref in record.refs if ref not in record_ids})
        if absent:
            missing[record_id] = absent
    return missing


def _limit_missing_reference_details(missing: dict[int, list[int]]) -> dict[str, object]:
    limited_items = list(sorted(missing.items()))[:DETAIL_MAX_MISSING_REFERENCE_RECORDS]
    return {
        "count": len(missing),
        "sample": {str(record_id): refs for record_id, refs in limited_items},
    }


def _reachable_entity_counts(
    records: dict[int, StepRecord],
    start_entity: str,
) -> dict[str, int]:
    start = start_entity.upper()
    stack = [record_id for record_id, record in records.items() if record.entity == start]
    seen: set[int] = set()
    counts: dict[str, int] = {}
    while stack:
        record_id = stack.pop()
        if record_id in seen:
            continue
        record = records.get(record_id)
        if record is None:
            continue
        seen.add(record_id)
        counts[record.entity] = counts.get(record.entity, 0) + 1
        stack.extend(ref for ref in record.refs if ref not in seen)
    return counts


def _occ_import_summary(path: Path) -> dict[str, Any]:
    try:
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.STEPControl import STEPControl_Reader
        from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID, TopAbs_VERTEX
        from OCP.TopExp import TopExp_Explorer
    except Exception as exc:
        return {
            "available": False,
            "read_ok": False,
            "error": f"OCP import unavailable: {type(exc).__name__}: {exc}",
        }

    try:
        reader = STEPControl_Reader()
        status = reader.ReadFile(str(path))
        if status != IFSelect_RetDone:
            return {
                "available": True,
                "read_ok": False,
                "error": f"OCC STEP reader returned status {status}",
            }
        transferred = reader.TransferRoots()
        shape = reader.OneShape()
    except Exception as exc:
        return {
            "available": True,
            "read_ok": False,
            "error": f"OCC STEP import failed: {type(exc).__name__}: {exc}",
        }

    def _count_topology(kind: Any) -> int:
        count = 0
        explorer = TopExp_Explorer(shape, kind)
        while explorer.More():
            count += 1
            explorer.Next()
        return count

    try:
        return {
            "available": True,
            "read_ok": True,
            "transferred_roots": int(transferred),
            "solid_count": _count_topology(TopAbs_SOLID),
            "face_count": _count_topology(TopAbs_FACE),
            "vertex_count": _count_topology(TopAbs_VERTEX),
        }
    except Exception as exc:
        return {
            "available": True,
            "read_ok": False,
            "error": f"OCC topology traversal failed: {type(exc).__name__}: {exc}",
        }


def _check_step(
    candidate_step: Path,
    reference_step: Path,
    reference_counts: dict[str, int],
) -> tuple[list[str], dict[str, Any]]:
    reasons: list[str] = []
    details: dict[str, Any] = {
        "candidate_size_bytes": candidate_step.stat().st_size,
        "reference_size_bytes": reference_step.stat().st_size,
    }
    candidate_records = _parse_step_records(candidate_step)
    reference_records = _parse_step_records(reference_step)
    details["candidate_step_record_count"] = len(candidate_records)
    details["reference_step_record_count"] = len(reference_records)

    head = candidate_step.open("rb").read(512).decode("utf-8", errors="ignore").upper()
    with candidate_step.open("rb") as handle:
        handle.seek(max(0, candidate_step.stat().st_size - 4096))
        tail = handle.read().decode("utf-8", errors="ignore").upper()

    if "ISO-10303-21" not in head:
        reasons.append("STEP file is missing ISO-10303-21 header")
    if "END-ISO-10303-21" not in tail:
        reasons.append("STEP file is missing END-ISO-10303-21 footer")

    min_size = int(reference_step.stat().st_size * STEP_MIN_REFERENCE_SIZE_FRACTION)
    if candidate_step.stat().st_size < min_size:
        reasons.append(
            f"STEP file is too small for a completed assembly: "
            f"{candidate_step.stat().st_size} bytes < {min_size} bytes"
        )

    reference_usage = _count_entity(reference_records, "NEXT_ASSEMBLY_USAGE_OCCURRENCE")
    candidate_usage = _count_entity(candidate_records, "NEXT_ASSEMBLY_USAGE_OCCURRENCE")
    candidate_usage_counts = _assembly_usage_counts(candidate_records)
    reference_usage_counts = _assembly_usage_counts(reference_records)
    candidate_geometry_counts = _geometry_entity_counts(candidate_records)
    reference_geometry_counts = _geometry_entity_counts(reference_records)
    candidate_occ = _occ_import_summary(candidate_step)
    reference_occ = _occ_import_summary(reference_step)
    candidate_reachable_geometry_counts = _reachable_entity_counts(
        candidate_records,
        "MANIFOLD_SOLID_BREP",
    )
    reference_reachable_geometry_counts = _reachable_entity_counts(
        reference_records,
        "MANIFOLD_SOLID_BREP",
    )
    candidate_missing_refs = _missing_references(
        candidate_records,
        (
            "NEXT_ASSEMBLY_USAGE_OCCURRENCE",
            "CONTEXT_DEPENDENT_SHAPE_REPRESENTATION",
            "MANIFOLD_SOLID_BREP",
            "CLOSED_SHELL",
            "ADVANCED_FACE",
            "FACE_OUTER_BOUND",
            "EDGE_LOOP",
            "ORIENTED_EDGE",
            "EDGE_CURVE",
            "VERTEX_POINT",
            "CARTESIAN_POINT",
            "PRODUCT_DEFINITION",
            "PRODUCT_DEFINITION_SHAPE",
            "SHAPE_DEFINITION_REPRESENTATION",
            "SHAPE_REPRESENTATION",
            "ADVANCED_BREP_SHAPE_REPRESENTATION",
        ),
    )
    details["candidate_next_assembly_usage_occurrence_count"] = candidate_usage
    details["reference_next_assembly_usage_occurrence_count"] = reference_usage
    details["candidate_assembly_usage_counts"] = candidate_usage_counts
    details["reference_assembly_usage_counts"] = reference_usage_counts
    details["candidate_geometry_entity_counts"] = candidate_geometry_counts
    details["reference_geometry_entity_counts"] = reference_geometry_counts
    details["candidate_occ_import"] = candidate_occ
    details["reference_occ_import"] = reference_occ
    details["candidate_reachable_from_solids_entity_counts"] = candidate_reachable_geometry_counts
    details["reference_reachable_from_solids_entity_counts"] = reference_reachable_geometry_counts
    details["candidate_missing_references_in_required_entities"] = _limit_missing_reference_details(
        candidate_missing_refs
    )
    if candidate_missing_refs:
        reasons.append("STEP required entity graph contains references to undefined records")

    if not reference_occ.get("read_ok"):
        reasons.append("Reference STEP failed OCC import validation; evaluator environment is incomplete")
    if not candidate_occ.get("read_ok"):
        reasons.append("Submitted STEP failed OCC import validation")
    if reference_occ.get("read_ok") and candidate_occ.get("read_ok"):
        reference_faces = int(reference_occ.get("face_count", 0))
        candidate_faces = int(candidate_occ.get("face_count", 0))
        reference_solids_occ = int(reference_occ.get("solid_count", 0))
        candidate_solids_occ = int(candidate_occ.get("solid_count", 0))
        if reference_faces > 0:
            min_faces = max(1, int(reference_faces * OCC_MIN_FACE_FRACTION))
            if candidate_faces < min_faces:
                reasons.append(
                    f"OCC imported face count is too low: {candidate_faces} < {min_faces}"
                )
        if reference_solids_occ > 0:
            min_solids_occ = max(1, int(reference_solids_occ * OCC_MIN_SOLID_FRACTION))
            if candidate_solids_occ < min_solids_occ:
                reasons.append(
                    f"OCC imported solid count is too low: {candidate_solids_occ} < {min_solids_occ}"
                )
    if reference_usage:
        min_usage = max(
            STEP_MIN_ASSEMBLY_USAGE_ABSOLUTE,
            int(reference_usage * STEP_MIN_ASSEMBLY_USAGE_FRACTION),
        )
        details["minimum_expected_assembly_usage_count"] = min_usage
        if candidate_usage < min_usage:
            reasons.append(
                f"STEP assembly instance count is too low: {candidate_usage} < {min_usage}"
            )

    if candidate_usage_counts != reference_counts:
        reasons.append("STEP assembly instance names/counts do not match the expected component set")
        details["assembly_usage_diff"] = {
            "missing_names": sorted(set(reference_counts) - set(candidate_usage_counts)),
            "extra_names": sorted(set(candidate_usage_counts) - set(reference_counts)),
            "wrong_quantities": {
                name: {
                    "candidate": candidate_usage_counts.get(name),
                    "reference": reference_counts.get(name),
                }
                for name in sorted(set(candidate_usage_counts) & set(reference_counts))
                if candidate_usage_counts[name] != reference_counts[name]
            },
        }

    for entity in ("CARTESIAN_POINT", "ADVANCED_FACE", "PRODUCT_DEFINITION"):
        reference_count = reference_geometry_counts[entity]
        if reference_count <= 0:
            continue
        minimum = max(1, int(reference_count * STEP_MIN_GEOMETRY_ENTITY_FRACTION))
        if candidate_geometry_counts[entity] < minimum:
            reasons.append(
                f"STEP geometry entity count for {entity} is too low: "
                f"{candidate_geometry_counts[entity]} < {minimum}"
            )

    for entity in ("CARTESIAN_POINT", "ADVANCED_FACE"):
        reference_count = reference_reachable_geometry_counts.get(entity, 0)
        if reference_count <= 0:
            continue
        minimum = max(1, int(reference_count * STEP_MIN_GEOMETRY_ENTITY_FRACTION))
        candidate_count = candidate_reachable_geometry_counts.get(entity, 0)
        if candidate_count < minimum:
            reasons.append(
                f"STEP connected solid geometry count for {entity} is too low: "
                f"{candidate_count} < {minimum}"
            )

    reference_solids = reference_geometry_counts["MANIFOLD_SOLID_BREP"]
    if reference_solids > 0:
        min_solids = max(1, int(reference_solids * STEP_MIN_MANIFOLD_SOLID_FRACTION))
        if candidate_geometry_counts["MANIFOLD_SOLID_BREP"] < min_solids:
            reasons.append(
                "STEP solid body count is too low: "
                f"{candidate_geometry_counts['MANIFOLD_SOLID_BREP']} < {min_solids}"
            )

    return reasons, details


def evaluate_submission(output_dir: Path, reference_dir: Path) -> dict[str, Any]:
    candidate_bom = output_dir / REQUIRED_BOM
    candidate_step = output_dir / REQUIRED_STEP
    reference_bom = reference_dir / REQUIRED_BOM
    reference_step = reference_dir / REQUIRED_STEP

    missing = [
        str(path)
        for path in (candidate_bom, candidate_step, reference_bom, reference_step)
        if not path.exists()
    ]
    if missing:
        return _result(0.0, [f"missing required files: {', '.join(missing)}"])

    reasons: list[str] = []
    details: dict[str, Any] = {}

    try:
        candidate_counts = _load_bom(candidate_bom)
        reference_counts = _load_bom(reference_bom)
    except Exception as exc:
        return _result(0.0, [f"failed to parse BOM: {exc}"])

    details["candidate_bom"] = candidate_counts
    details["reference_bom"] = reference_counts
    if candidate_counts != reference_counts:
        reasons.append("BOM component names/quantities do not match the reference")
        missing_names = sorted(set(reference_counts) - set(candidate_counts))
        extra_names = sorted(set(candidate_counts) - set(reference_counts))
        wrong_quantities = {
            name: {
                "candidate": candidate_counts.get(name),
                "reference": reference_counts.get(name),
            }
            for name in sorted(set(candidate_counts) & set(reference_counts))
            if candidate_counts[name] != reference_counts[name]
        }
        details["bom_diff"] = {
            "missing_names": missing_names,
            "extra_names": extra_names,
            "wrong_quantities": wrong_quantities,
        }

    step_reasons, step_details = _check_step(candidate_step, reference_step, reference_counts)
    details["step"] = step_details
    reasons.extend(step_reasons)

    return _result(1.0 if not reasons else 0.0, reasons, details)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = evaluate_submission(Path(args.output_dir), Path(args.reference_dir))
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
