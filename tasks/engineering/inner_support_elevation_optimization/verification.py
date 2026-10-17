"""Source integrity and report syntax checks; no FE compliance or scoring."""

import csv
import hashlib
import io
import json
import math
from pathlib import Path

CASES = tuple(f"pos_{position}m" for position in range(5))
RESPONSES = tuple(
    f"{case}_max_{quantity}_mm" for quantity in ("settlement", "disp") for case in CASES
)
DERIVED = (
    "reduction_0m_to_2m_percent",
    "increase_2m_to_4m_percent",
    "best_support_position_for_minimum_settlement_m",
    "best_support_position_for_minimum_disp_m",
)
REQUIRED_IMAGES = (
    "pos_0m_output_view.png",
    "pos_2m_output_view.png",
    "pos_4m_output_view.png",
    "settlement_vs_support_position.png",
    "lateral_disp_vs_support_position.png",
)
REPLAY_UNAVAILABLE = (
    "Independent five-case native replay is unavailable. This is an "
    "evaluator error, not a solver score. See assets/native_task_integration.md."
)


class EvaluationUnavailableError(RuntimeError):
    """The evaluator cannot establish a valid engineering score."""


class SourceBundleError(RuntimeError):
    """The prepared task input differs from the audited original sources."""


def verify_source_bundle(files):
    """Check original filename-to-bytes entries; this does not validate a model."""
    manifest = json.loads((Path(__file__).with_name("assets") / "source_manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        content = files.get(name)
        if not isinstance(content, bytes):
            raise SourceBundleError(f"Missing original source: {name}")
        if hashlib.sha256(content).hexdigest() != expected:
            raise SourceBundleError(f"Original source changed: {name}")


def parse_json(content):
    def unique_pairs(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise ValueError(f"Duplicate JSON member: {name}")
            result[name] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    return json.loads(content, object_pairs_hook=unique_pairs, parse_constant=invalid_constant)


def read_answer(content):
    answer = parse_json(content)
    if not isinstance(answer, dict) or not set(RESPONSES + DERIVED).issubset(answer):
        raise ValueError("Answer must contain all fourteen required fields")
    for name in RESPONSES + DERIVED:
        value = answer[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"Expected finite JSON number for {name}")
    if any(answer[name] <= 0 for name in RESPONSES):
        raise ValueError("Response magnitudes must be positive")
    if any(answer[name] not in range(5) for name in DERIVED[2:]):
        raise ValueError("Best positions must be integers from 0 through 4")
    return answer


def read_csv(content, answer):
    """Check table/report consistency without establishing physical correctness."""
    reader = csv.DictReader(io.StringIO(content))
    required = {"case_label", "support_position_m", "max_settlement_mm", "max_disp_mm"}
    if (
        not reader.fieldnames
        or len(set(reader.fieldnames)) != len(reader.fieldnames)
        or not required.issubset(reader.fieldnames)
    ):
        raise ValueError("Missing or duplicate case-summary columns")
    rows = list(reader)
    if len(rows) != 5 or {row["case_label"] for row in rows} != set(CASES):
        raise ValueError("Each case must occur exactly once")
    for row in rows:
        case = row["case_label"]
        if None in row or any(row[name] is None for name in required):
            raise ValueError("Missing or extra unnamed cells")
        if float(row["support_position_m"]) != CASES.index(case):
            raise ValueError("Invalid case position")
        for quantity in ("settlement", "disp"):
            value = float(row[f"max_{quantity}_mm"])
            if not math.isfinite(value) or value != answer[f"{case}_max_{quantity}_mm"]:
                raise ValueError("CSV response disagrees with answer.json")
    return rows
