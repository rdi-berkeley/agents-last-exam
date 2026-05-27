"""Local scoring helpers for sagebrush_ridge_uav_survey_pack."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from typing import Any


def read_csv_text(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def load_json_text(text: str) -> dict[str, Any]:
    return json.loads(text)


def plan_signature(plan_text: str) -> list[dict[str, str]]:
    data = load_json_text(plan_text)
    mission = data.get("mission")
    if not isinstance(mission, dict):
        raise ValueError("missing mission object")
    rows: list[dict[str, str]] = []
    items = mission.get("items")
    if not isinstance(items, list):
        raise ValueError("missing mission.items list")
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("mission.items must contain objects")
        command = item.get("command")
        params = item.get("params", [])
        if command in {16, 22}:
            if not isinstance(params, list) or len(params) < 7:
                raise ValueError(f"command {command} missing waypoint params")
            rows.append(
                {
                    "command": str(command),
                    "lat": f"{float(params[4]):.7f}",
                    "lon": f"{float(params[5]):.7f}",
                    "alt": f"{float(params[6]):.1f}",
                }
            )
        elif command in {20, 206}:
            rows.append({"command": str(command), "lat": "", "lon": "", "alt": ""})
    return rows


def compare_rows(
    actual: list[dict[str, Any]],
    expected: list[dict[str, Any]],
    numeric_fields: dict[str, float],
) -> tuple[bool, str]:
    if len(actual) != len(expected):
        return False, f"row_count {len(actual)} != {len(expected)}"
    for idx, (a_row, e_row) in enumerate(zip(actual, expected), start=1):
        if set(a_row) != set(e_row):
            missing = sorted(set(e_row) - set(a_row))
            extra = sorted(set(a_row) - set(e_row))
            return False, f"row {idx} keys mismatch missing={missing} extra={extra}"
        for key, e_value in e_row.items():
            a_value = a_row.get(key)
            if key in numeric_fields and str(e_value) != "":
                try:
                    if abs(float(a_value) - float(e_value)) > numeric_fields[key]:
                        return False, f"row {idx} {key}: {a_value} != {e_value}"
                except (TypeError, ValueError):
                    return False, f"row {idx} {key}: {a_value} is not numeric"
            elif str(a_value) != str(e_value):
                return False, f"row {idx} {key}: {a_value} != {e_value}"
    return True, ""


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def compare_json_dict(
    actual: dict[str, Any],
    expected: dict[str, Any],
    numeric_tolerance: float,
) -> tuple[bool, str]:
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        return False, "json payload must be an object"
    actual_keys = set(actual)
    expected_keys = set(expected)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        return False, f"keys mismatch missing={missing} extra={extra}"

    for key in sorted(expected):
        actual_value = actual.get(key)
        expected_value = expected[key]
        if _is_number(expected_value):
            try:
                if abs(float(actual_value) - float(expected_value)) > numeric_tolerance:
                    return False, f"{key}: {actual_value} != {expected_value}"
            except (TypeError, ValueError):
                return False, f"{key}: {actual_value} is not numeric"
        elif actual_value != expected_value:
            return False, f"{key}: {actual_value} != {expected_value}"
    return True, ""


@dataclass
class ScenarioReport:
    scenario_id: str
    score: float
    plan_match: bool
    manifest_match: bool
    safety_match: bool
    details: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "score": self.score,
            "plan_match": self.plan_match,
            "manifest_match": self.manifest_match,
            "safety_match": self.safety_match,
            "details": self.details,
        }


@dataclass
class BundleScore:
    total_points: float
    scenario_reports: list[ScenarioReport]


def score_bundle_from_texts(
    *,
    scenario_texts: dict[str, str],
    output_texts: dict[str, str],
    reference_texts: dict[str, str],
) -> BundleScore:
    reports: list[ScenarioReport] = []
    total_points = 0.0
    for scenario_filename in sorted(scenario_texts):
        scenario = load_json_text(scenario_texts[scenario_filename])
        scenario_id = scenario["scenario_id"]
        plan_name = f"{scenario_id}.plan"
        manifest_name = f"{scenario_id}_manifest.csv"
        safety_name = f"{scenario_id}_safety_report.json"

        plan_match = False
        manifest_match = False
        safety_match = False
        details: list[str] = []
        scenario_points = 0.0

        actual_plan = output_texts.get(plan_name)
        expected_plan = reference_texts.get(plan_name)
        if actual_plan is None or expected_plan is None:
            details.append("missing plan")
        else:
            try:
                actual_plan_rows = plan_signature(actual_plan)
                expected_plan_rows = plan_signature(expected_plan)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                plan_match = False
                plan_detail = f"invalid plan: {exc}"
            else:
                plan_match, plan_detail = compare_rows(
                    actual_plan_rows,
                    expected_plan_rows,
                    {"lat": 0.0000001, "lon": 0.0000001, "alt": 0.1},
                )
            details.append(plan_detail)
            if plan_match:
                scenario_points += 18.0

        actual_manifest = output_texts.get(manifest_name)
        expected_manifest = reference_texts.get(manifest_name)
        if actual_manifest is None or expected_manifest is None:
            details.append("missing manifest")
        else:
            manifest_match, manifest_detail = compare_rows(
                read_csv_text(actual_manifest),
                read_csv_text(expected_manifest),
                {"transects": 0.0, "distance_m": 0.1, "trigger_count": 0.0},
            )
            details.append(manifest_detail)
            if manifest_match:
                scenario_points += 13.5

        actual_safety = output_texts.get(safety_name)
        expected_safety = reference_texts.get(safety_name)
        if actual_safety is None or expected_safety is None:
            details.append("missing safety_report")
        else:
            try:
                actual_safety_json = load_json_text(actual_safety)
                expected_safety_json = load_json_text(expected_safety)
            except json.JSONDecodeError as exc:
                safety_match = False
                safety_detail = f"invalid safety_report: {exc.msg}"
            else:
                safety_match, safety_detail = compare_json_dict(
                    actual_safety_json,
                    expected_safety_json,
                    numeric_tolerance=0.1,
                )
            details.append(safety_detail)
            if safety_match:
                scenario_points += 13.5

        total_points += scenario_points
        reports.append(
            ScenarioReport(
                scenario_id=scenario_id,
                score=scenario_points,
                plan_match=plan_match,
                manifest_match=manifest_match,
                safety_match=safety_match,
                details=details,
            )
        )
    return BundleScore(total_points=total_points, scenario_reports=reports)


def normalize_bundle_score(total_points: float, scenario_count: int) -> float:
    max_points = 45.0 * scenario_count
    return total_points / max_points if max_points else 0.0


async def load_scenario_texts(session, scenario_dir: str) -> dict[str, str]:
    scenario_texts: dict[str, str] = {}
    for filename in await session.list_dir(scenario_dir):
        if not filename.endswith(".json"):
            continue
        remote_path = f"{scenario_dir}/{filename}"
        text = await session.read_file(remote_path)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if "scenario_id" in payload:
            scenario_texts[filename] = text
    return scenario_texts
