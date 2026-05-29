#!/usr/bin/env python
"""Score DC24_0406 PCB CAD assembly outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


EXPECTED_STEP = "DC24_0406_assembled.step"
EXPECTED_BOM = "DC24_0406_BOM.csv"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_name(value: str) -> str:
    name = value.strip().strip('"').strip("'")
    if name.lower().endswith(".step"):
        name = name[:-5]
    return name


def _read_bom(path: Path) -> dict[str, int]:
    rows: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("BOM has no header")
        lower_to_original = {name.lower().strip(): name for name in reader.fieldnames}
        name_key = lower_to_original.get("name") or lower_to_original.get("part") or lower_to_original.get("component")
        qty_key = lower_to_original.get("quantity") or lower_to_original.get("qty") or lower_to_original.get("count")
        if not name_key or not qty_key:
            raise ValueError(f"BOM must contain name and quantity columns; found {reader.fieldnames}")
        for row in reader:
            name = _normalize_name(row.get(name_key, ""))
            if not name:
                continue
            try:
                qty = int(float((row.get(qty_key) or "0").strip()))
            except ValueError as exc:
                raise ValueError(f"invalid quantity for {name!r}") from exc
            rows[name] = rows.get(name, 0) + qty
    return rows


def _load_targets(reference_dir: Path) -> dict:
    targets_path = reference_dir / "objective_targets.json"
    if targets_path.exists():
        return json.loads(targets_path.read_text(encoding="utf-8"))
    return {
        "expected_bom": _read_bom(reference_dir / EXPECTED_BOM),
        "reference_step_occurrences": _step_occurrences((reference_dir / EXPECTED_STEP).read_text(errors="ignore")),
    }


def _step_occurrences(text: str) -> dict[str, int]:
    names = re.findall(r"NEXT_ASSEMBLY_USAGE_OCCURRENCE\('[^']*','([^']+)'", text)
    counts: dict[str, int] = {}
    for raw_name in names:
        name = _normalize_name(raw_name)
        counts[name] = counts.get(name, 0) + 1
    return counts


def _step_translations(text: str) -> dict[str, list[list[float]]]:
    flat = " ".join(text.split())
    entities = {
        f"#{match.group(1)}": match.group(2)
        for match in re.finditer(r"#(\d+)=(.*?);(?=\s*#\d+=|\s*ENDSEC)", flat)
    }
    nauo_names = {
        f"#{match.group(1)}": _normalize_name(match.group(2))
        for match in re.finditer(
            r"#(\d+)=NEXT_ASSEMBLY_USAGE_OCCURRENCE\('[^']*','([^']+)'",
            flat,
        )
    }
    product_shape_to_nauo = {
        f"#{match.group(1)}": f"#{match.group(2)}"
        for match in re.finditer(
            r"#(\d+)=PRODUCT_DEFINITION_SHAPE\('[^']*','NAUO PRDDFN',#(\d+)\)",
            flat,
        )
    }
    relationship_to_shape = {
        match.group(2): match.group(3)
        for match in re.finditer(
            r"#(\d+)=CONTEXT_DEPENDENT_SHAPE_REPRESENTATION\((#\d+),(#\d+)\)",
            flat,
        )
    }
    relationship_to_transform = {
        entity_id: match.group(1)
        for entity_id, body in entities.items()
        if (match := re.search(r"REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION\((#\d+)\)", body))
    }
    transform_to_placement = {
        f"#{match.group(1)}": match.group(3)
        for match in re.finditer(
            r"#(\d+)=ITEM_DEFINED_TRANSFORMATION\('[^']*','[^']*',(#\d+),(#\d+)\)",
            flat,
        )
    }
    placement_to_point = {
        f"#{match.group(1)}": match.group(2)
        for match in re.finditer(
            r"#(\d+)=AXIS2_PLACEMENT_3D\('[^']*',(#\d+),#\d+,#\d+\)",
            flat,
        )
    }
    point_coords = {
        f"#{match.group(1)}": [float(match.group(2)), float(match.group(3)), float(match.group(4))]
        for match in re.finditer(
            r"#(\d+)=CARTESIAN_POINT\('[^']*',\(([-+0-9.Ee]+),([-+0-9.Ee]+),([-+0-9.Ee]+)\)\)",
            flat,
        )
    }

    translations: dict[str, list[list[float]]] = {}
    for relationship, shape in relationship_to_shape.items():
        nauo = product_shape_to_nauo.get(shape)
        name = nauo_names.get(nauo or "")
        transform = relationship_to_transform.get(relationship)
        placement = transform_to_placement.get(transform or "")
        point = placement_to_point.get(placement or "")
        coords = point_coords.get(point or "")
        if not name or coords is None:
            continue
        translations.setdefault(name, []).append(coords)
    return {name: sorted(coords) for name, coords in translations.items()}


def _named_products(text: str) -> set[str]:
    names = set()
    for raw_name in re.findall(r"PRODUCT\('([^']+)'", text):
        names.add(_normalize_name(raw_name))
    for raw_name in re.findall(r"'([^']+\.step)'", text, flags=re.IGNORECASE):
        names.add(_normalize_name(raw_name))
    return names


def _score_step(step_path: Path, reference_dir: Path, targets: dict, input_dir: Path | None) -> tuple[list[str], list[str]]:
    hard_failures: list[str] = []
    details: list[str] = []
    step_bytes = step_path.read_bytes()
    step_text = step_bytes.decode("utf-8", errors="ignore")
    ref_step = reference_dir / EXPECTED_STEP
    ref_size = ref_step.stat().st_size

    if not step_text.lstrip().startswith("ISO-10303-21"):
        hard_failures.append("STEP output is not an ISO-10303-21 file")
        return hard_failures, details

    if input_dir:
        board_path = input_dir / "parts" / "Board.step"
        if board_path.exists() and _sha256(step_path) == _sha256(board_path):
            hard_failures.append("STEP output is the bare board input, not a completed assembly")

    if len(step_bytes) < int(ref_size * 0.55):
        hard_failures.append("STEP output is much smaller than the reference assembly")

    expected_occurrences = {str(k): int(v) for k, v in targets["reference_step_occurrences"].items()}
    observed_occurrences = _step_occurrences(step_text)
    if observed_occurrences:
        wrong_counts = {
            name: observed_occurrences.get(name, 0)
            for name, expected in expected_occurrences.items()
            if observed_occurrences.get(name, 0) != expected
        }
        if wrong_counts:
            hard_failures.append("STEP assembly occurrence counts do not match the expected component set")
            details.append(f"observed_occurrences={observed_occurrences}")
        unexpected_occurrences = sorted(set(observed_occurrences) - set(expected_occurrences))
        if unexpected_occurrences:
            details.append(f"unexpected_occurrence_names={unexpected_occurrences}")

        expected_translations = targets.get("reference_step_translations") or {}
        observed_translations = _step_translations(step_text)
        if expected_translations and observed_translations:
            mismatched = []
            tolerance_m = 0.003
            for name, expected_points in expected_translations.items():
                observed_points = observed_translations.get(name)
                if observed_points is None or len(observed_points) != len(expected_points):
                    mismatched.append(name)
                    continue
                for expected, observed in zip(sorted(expected_points), sorted(observed_points)):
                    distance = sum((float(a) - float(b)) ** 2 for a, b in zip(expected, observed)) ** 0.5
                    if distance > tolerance_m:
                        mismatched.append(name)
                        break
            if mismatched:
                hard_failures.append("STEP assembly component placements differ from the reference layout")
                details.append(f"placement_mismatches={sorted(set(mismatched))}")
        elif expected_translations:
            details.append("STEP occurrence records found but placement translations could not be parsed")
    else:
        details.append("STEP output has no assembly occurrence records; using fallback product-name checks")
        product_names = _named_products(step_text)
        expected_names = set(expected_occurrences)
        missing = sorted(expected_names - product_names)
        if len(missing) > 1:
            hard_failures.append("STEP output is missing multiple expected component names")
            details.append(f"missing_component_names={missing}")
        major_names = {
            "Board",
            "A-TB500-TO02",
            "CAP_AL_16.0x26.0",
            "Radiator_Big",
            "Trans",
            "TO-220-3",
        }
        missing_major = sorted(major_names - product_names)
        if missing_major:
            hard_failures.append("STEP output is missing distinctive major components")
            details.append(f"missing_major_components={missing_major}")

    return hard_failures, details


def score_submission(submission_dir: Path, reference_dir: Path, input_dir: Path | None = None) -> dict:
    hard_failures: list[str] = []
    details: list[str] = []
    targets = _load_targets(reference_dir)

    step_path = submission_dir / EXPECTED_STEP
    bom_path = submission_dir / EXPECTED_BOM
    if not step_path.exists():
        hard_failures.append(f"Missing required output: {EXPECTED_STEP}")
    if not bom_path.exists():
        hard_failures.append(f"Missing required output: {EXPECTED_BOM}")
    if hard_failures:
        return {
            "score": 0,
            "final_score": 0.0,
            "passed": False,
            "hard_failures": hard_failures,
            "details": details,
        }

    try:
        observed_bom = _read_bom(bom_path)
    except Exception as exc:
        hard_failures.append(f"Could not parse BOM: {exc}")
        observed_bom = {}

    expected_bom = {str(k): int(v) for k, v in targets["expected_bom"].items()}
    if observed_bom != expected_bom:
        hard_failures.append("BOM quantities do not exactly match the expected flattened BOM")
        details.append(f"observed_bom={observed_bom}")

    step_failures, step_details = _score_step(step_path, reference_dir, targets, input_dir)
    hard_failures.extend(step_failures)
    details.extend(step_details)

    passed = not hard_failures
    return {
        "score": 100 if passed else 0,
        "final_score": 1.0 if passed else 0.0,
        "passed": passed,
        "hard_failures": hard_failures,
        "details": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--input-dir")
    args = parser.parse_args()
    result = score_submission(
        Path(args.submission_dir),
        Path(args.reference_dir),
        Path(args.input_dir) if args.input_dir else None,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
