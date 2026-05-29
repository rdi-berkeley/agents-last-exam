from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = ["panel_id", "quantity_pct", "uniformity_pct", "composite_pct"]
REQUIRED_PANEL_IDS = ["panel_sanded_01", "panel_sanded_02", "panel_sanded_03"]
METRIC_COLUMNS = ["quantity_pct", "uniformity_pct", "composite_pct"]
TOLERANCE = 1.0


def _read_csv_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig", errors="replace")


def _parse_csv(text: str) -> tuple[list[dict[str, str]], str | None]:
    try:
        reader = csv.DictReader(io.StringIO(text))
        if reader.fieldnames != REQUIRED_COLUMNS:
            return [], f"expected columns {REQUIRED_COLUMNS}, got {reader.fieldnames}"
        rows = list(reader)
    except csv.Error as exc:
        return [], f"csv parse error: {exc}"
    return rows, None


def score_csv_text(candidate_text: str, reference_text: str) -> dict[str, Any]:
    candidate_rows, candidate_error = _parse_csv(candidate_text)
    if candidate_error:
        return {"score": 0.0, "hard_gate": candidate_error}
    reference_rows, reference_error = _parse_csv(reference_text)
    if reference_error:
        return {"score": 0.0, "hard_gate": f"invalid reference: {reference_error}"}

    candidate_by_panel = {row["panel_id"]: row for row in candidate_rows}
    reference_by_panel = {row["panel_id"]: row for row in reference_rows}
    if len(candidate_by_panel) != len(candidate_rows):
        return {"score": 0.0, "hard_gate": "duplicate panel_id in candidate"}
    if list(reference_by_panel) != REQUIRED_PANEL_IDS:
        return {"score": 0.0, "hard_gate": "reference panel ids are invalid"}

    missing = [panel_id for panel_id in REQUIRED_PANEL_IDS if panel_id not in candidate_by_panel]
    extra = [panel_id for panel_id in candidate_by_panel if panel_id not in REQUIRED_PANEL_IDS]
    if missing or extra:
        return {"score": 0.0, "hard_gate": {"missing_panel_ids": missing, "extra_panel_ids": extra}}

    checks: list[dict[str, Any]] = []
    passed = 0
    for panel_id in REQUIRED_PANEL_IDS:
        candidate = candidate_by_panel[panel_id]
        reference = reference_by_panel[panel_id]
        for metric in METRIC_COLUMNS:
            try:
                candidate_value = float(candidate[metric])
                reference_value = float(reference[metric])
            except (TypeError, ValueError):
                return {
                    "score": 0.0,
                    "hard_gate": f"non-numeric value for {panel_id}/{metric}",
                }
            abs_error = abs(candidate_value - reference_value)
            ok = abs_error <= TOLERANCE
            passed += int(ok)
            checks.append(
                {
                    "panel_id": panel_id,
                    "metric": metric,
                    "candidate": candidate_value,
                    "reference": reference_value,
                    "abs_error": abs_error,
                    "passed": ok,
                }
            )

    total = len(REQUIRED_PANEL_IDS) * len(METRIC_COLUMNS)
    return {
        "score": passed / total,
        "passed_checks": passed,
        "total_checks": total,
        "tolerance": TOLERANCE,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference", required=True)
    args = parser.parse_args()

    candidate_path = Path(args.candidate)
    reference_path = Path(args.reference)
    if not candidate_path.exists():
        print(json.dumps({"score": 0.0, "hard_gate": "candidate file missing"}, indent=2))
        return 0
    if not reference_path.exists():
        print(json.dumps({"score": 0.0, "hard_gate": "reference file missing"}, indent=2))
        return 0
    print(
        json.dumps(
            score_csv_text(_read_csv_text(candidate_path), _read_csv_text(reference_path)),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
