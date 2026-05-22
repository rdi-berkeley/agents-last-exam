"""Local scorer for humanoid_wbc_policy_evaluation outputs."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VERDICTS = {"successful", "nearly_successful", "failed"}
LABEL_WEIGHT = 0.7
EVIDENCE_WEIGHT = 0.3
REQUIRED_TOP_LEVEL = {"task_id", "evaluations", "summary"}
REQUIRED_EVAL_FIELDS = {
    "case_id",
    "motion",
    "mjlab_task",
    "motion_file",
    "checkpoint_file",
    "verdict",
    "confidence",
    "evidence",
}
REQUIRED_SUMMARY_FIELDS = {"successful", "nearly_successful", "failed", "overall_notes"}
ALLOWED_VISUAL_SUFFIXES = {".gif", ".html", ".mp4", ".webm"}
MIN_VISUAL_DEMO_BYTES = 512
EVIDENCE_CUES = {
    "successful": {
        "smooth",
        "stable",
        "stably",
        "identical",
        "close",
        "reference",
        "tracks",
        "sits",
    },
    "nearly_successful": {
        "unstable",
        "unstably",
        "delay",
        "delayed",
        "behind",
        "last",
        "late",
        "imperfect",
        "nearly",
        "does not successfully",
        "fails to keep up",
    },
    "failed": {
        "fall",
        "falls",
        "fell",
        "slides",
        "sideways",
        "immediately",
        "disordered",
        "unrecoverable",
        "cannot",
        "fails",
    },
}


@dataclass
class ScoreResult:
    score: float
    diagnostics: list[str]


def _load_json(path: Path) -> tuple[Any | None, list[str]]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), []
    except FileNotFoundError:
        return None, [f"missing file: {path.name}"]
    except json.JSONDecodeError as exc:
        return None, [f"invalid JSON: {exc}"]


def _validate_structure(report: Any, expected_case_ids: set[str]) -> list[str]:
    errors: list[str] = []
    if not isinstance(report, dict):
        return ["top-level report must be an object"]
    if set(report) != REQUIRED_TOP_LEVEL:
        errors.append(f"top-level keys must be exactly {sorted(REQUIRED_TOP_LEVEL)}")
    if report.get("task_id") != "humanoid_wbc_policy_evaluation":
        errors.append("task_id must be humanoid_wbc_policy_evaluation")

    evaluations = report.get("evaluations")
    if not isinstance(evaluations, list):
        errors.append("evaluations must be an array")
        evaluations = []
    if len(evaluations) != len(expected_case_ids):
        errors.append(f"evaluations must contain exactly {len(expected_case_ids)} items")

    seen: list[str] = []
    for idx, item in enumerate(evaluations):
        if not isinstance(item, dict):
            errors.append(f"evaluations[{idx}] must be an object")
            continue
        if set(item) != REQUIRED_EVAL_FIELDS:
            errors.append(f"evaluations[{idx}] keys must be exactly {sorted(REQUIRED_EVAL_FIELDS)}")
        case_id = item.get("case_id")
        if not isinstance(case_id, str):
            errors.append(f"evaluations[{idx}].case_id must be a string")
        else:
            seen.append(case_id)
            if case_id not in expected_case_ids:
                errors.append(f"unexpected case_id: {case_id}")
        if item.get("verdict") not in VERDICTS:
            errors.append(f"evaluations[{idx}].verdict must be one of {sorted(VERDICTS)}")
        confidence = item.get("confidence")
        if not isinstance(confidence, int | float) or not 0 <= confidence <= 1:
            errors.append(f"evaluations[{idx}].confidence must be a number in [0, 1]")
        evidence = item.get("evidence")
        if not isinstance(evidence, dict):
            errors.append(f"evaluations[{idx}].evidence must be an object")
        else:
            observation = evidence.get("observation")
            if set(evidence) - {"observation", "visual_demo_path", "notes"}:
                errors.append(f"evaluations[{idx}].evidence has unsupported keys")
            if not isinstance(observation, str) or len(observation.strip()) < 20:
                errors.append(f"evaluations[{idx}].evidence.observation must be at least 20 characters")
            demo_path = evidence.get("visual_demo_path")
            if not isinstance(demo_path, str) or not demo_path.startswith("visual_demos/"):
                errors.append(
                    f"evaluations[{idx}].evidence.visual_demo_path must point under visual_demos/"
                )
            elif Path(demo_path).suffix.lower() not in ALLOWED_VISUAL_SUFFIXES:
                errors.append(
                    f"evaluations[{idx}].evidence.visual_demo_path must end with one of "
                    f"{sorted(ALLOWED_VISUAL_SUFFIXES)}"
                )

    counts = Counter(seen)
    duplicates = sorted(case_id for case_id, count in counts.items() if count > 1)
    missing = sorted(expected_case_ids - set(seen))
    if duplicates:
        errors.append(f"duplicate case_id values: {duplicates}")
    if missing:
        errors.append(f"missing case_id values: {missing}")

    summary = report.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
    else:
        if set(summary) != REQUIRED_SUMMARY_FIELDS:
            errors.append(f"summary keys must be exactly {sorted(REQUIRED_SUMMARY_FIELDS)}")
        observed_summary = Counter(
            item.get("verdict")
            for item in evaluations
            if isinstance(item, dict) and item.get("verdict") in VERDICTS
        )
        for verdict in sorted(VERDICTS):
            if summary.get(verdict) != observed_summary[verdict]:
                errors.append(f"summary.{verdict} must equal the number of evaluation items with that verdict")
        notes = summary.get("overall_notes")
        if not isinstance(notes, str) or len(notes.strip()) < 20:
            errors.append("summary.overall_notes must be at least 20 characters")

    return errors


def _evidence_matches_expected_behavior(observation: str, expected_verdict: str) -> bool:
    normalized = " ".join(observation.lower().split())
    if len(normalized) < 40:
        return False
    return any(cue in normalized for cue in EVIDENCE_CUES[expected_verdict])


def _visual_demo_errors(report: dict[str, Any], output_dir: Path) -> list[str]:
    errors: list[str] = []
    for item in report["evaluations"]:
        case_id = item["case_id"]
        demo_path = Path(item["evidence"]["visual_demo_path"])
        if demo_path.is_absolute() or ".." in demo_path.parts:
            errors.append(f"{case_id}: visual_demo_path must be a safe relative path")
            continue
        if demo_path.parts[0] != "visual_demos":
            errors.append(f"{case_id}: visual_demo_path must start with visual_demos/")
            continue
        demo_file = output_dir / demo_path
        if not demo_file.exists():
            errors.append(f"{case_id}: missing visual demo file {demo_path}")
            continue
        if not demo_file.is_file():
            errors.append(f"{case_id}: visual demo path is not a file {demo_path}")
            continue
        if demo_file.suffix.lower() not in ALLOWED_VISUAL_SUFFIXES:
            errors.append(f"{case_id}: unsupported visual demo extension {demo_file.suffix}")
            continue
        size = demo_file.stat().st_size
        if size < MIN_VISUAL_DEMO_BYTES:
            errors.append(
                f"{case_id}: visual demo {demo_path} is too small "
                f"({size} bytes < {MIN_VISUAL_DEMO_BYTES})"
            )
    return errors


def score_report(report_path: Path, reference_path: Path, output_dir: Path | None = None) -> ScoreResult:
    reference, ref_errors = _load_json(reference_path)
    if ref_errors:
        return ScoreResult(0.0, ref_errors)
    if not isinstance(reference, dict) or not isinstance(reference.get("cases"), list):
        return ScoreResult(0.0, ["reference expected_verdicts.json is malformed"])

    expected_verdicts = {
        item["case_id"]: item["expected_verdict"]
        for item in reference["cases"]
        if isinstance(item, dict) and "case_id" in item and "expected_verdict" in item
    }
    expected_observations = {
        item["case_id"]: item.get("gold_observation", "")
        for item in reference["cases"]
        if isinstance(item, dict) and "case_id" in item
    }
    if len(expected_verdicts) != len(reference["cases"]):
        return ScoreResult(0.0, ["reference contains malformed case records"])

    report, report_errors = _load_json(report_path)
    if report_errors:
        return ScoreResult(0.0, report_errors)

    structure_errors = _validate_structure(report, set(expected_verdicts))
    if structure_errors:
        return ScoreResult(0.0, structure_errors)
    if output_dir is None:
        output_dir = report_path.parent
    visual_errors = _visual_demo_errors(report, output_dir)
    if visual_errors:
        return ScoreResult(0.0, visual_errors)

    evaluations = report["evaluations"]
    correct = sum(
        1 for item in evaluations if item["verdict"] == expected_verdicts[item["case_id"]]
    )
    label_score = correct / len(expected_verdicts)

    evidence_count = sum(
        1
        for item in evaluations
        if _evidence_matches_expected_behavior(
            item["evidence"]["observation"],
            expected_verdicts[item["case_id"]],
        )
    )
    evidence_score = evidence_count / len(expected_verdicts)

    score = LABEL_WEIGHT * label_score + EVIDENCE_WEIGHT * evidence_score
    diagnostics = [
        f"correct_labels={correct}/{len(expected_verdicts)}",
        f"evidence_items={evidence_count}/{len(expected_verdicts)}",
        f"weights=label:{LABEL_WEIGHT},evidence:{EVIDENCE_WEIGHT}",
    ]
    if evidence_count < len(expected_verdicts):
        weak_case_ids = [
            item["case_id"]
            for item in evaluations
            if not _evidence_matches_expected_behavior(
                item["evidence"]["observation"],
                expected_verdicts[item["case_id"]],
            )
        ]
        diagnostics.append(f"weak_evidence_case_ids={weak_case_ids}")
        diagnostics.append(
            "evidence is checked for minimum length plus verdict-specific behavior cues; "
            f"hidden gold observations are available for audit: {bool(expected_observations)}"
        )
    return ScoreResult(round(float(score), 6), diagnostics)
