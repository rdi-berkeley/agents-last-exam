#!/usr/bin/env python3
"""Score a candidate grading submission against hidden gold tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REQUIRED_OUTPUT_FILES = [
    "grades.csv",
    "error_tags.csv",
    "per_student_feedback.json",
    "common_mistakes_summary.md",
    "grader_manifest.json",
]
REQUIRED_MANIFEST_KEYS = ["python_version", "platform", "rubric_version"]
GRADE_FIELDS = ["problem_1a", "problem_1b", "problem_2a", "problem_2b", "total_score"]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _lower_contains(text: str, phrase: str) -> bool:
    return phrase.lower() in text.lower()


def score_submission(*, submission_dir: Path, reference_dir: Path) -> dict[str, object]:
    payload: dict[str, object] = {
        "score": 0.0,
        "passed": False,
        "required_output_files": REQUIRED_OUTPUT_FILES,
    }

    missing_files = [name for name in REQUIRED_OUTPUT_FILES if not (submission_dir / name).exists()]
    if missing_files:
        payload["error"] = "missing required output files"
        payload["missing_files"] = missing_files
        return payload

    try:
        candidate_scores = _read_csv(submission_dir / "grades.csv")
        gold_scores = _read_csv(reference_dir / "gold_scores.csv")
        candidate_tags = _read_csv(submission_dir / "error_tags.csv")
        gold_tags = _read_csv(reference_dir / "gold_error_tags.csv")
        candidate_feedback = json.loads((submission_dir / "per_student_feedback.json").read_text(encoding="utf-8"))
        candidate_manifest = json.loads((submission_dir / "grader_manifest.json").read_text(encoding="utf-8"))
        feedback_requirements = json.loads(
            (reference_dir / "feedback_requirements.json").read_text(encoding="utf-8")
        )
        summary_requirements = json.loads(
            (reference_dir / "summary_requirements.json").read_text(encoding="utf-8")
        )
        candidate_summary = (submission_dir / "common_mistakes_summary.md").read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        payload["error"] = f"failed to parse candidate outputs: {exc}"
        return payload

    gold_score_map = {row["student_id"]: row for row in gold_scores}
    candidate_score_map = {row["student_id"]: row for row in candidate_scores}
    score_cell_matches = 0
    total_score_cells = len(gold_score_map) * len(GRADE_FIELDS)
    for student_id, gold_row in gold_score_map.items():
        candidate_row = candidate_score_map.get(student_id)
        if candidate_row is None:
            continue
        for field in GRADE_FIELDS:
            try:
                if abs(float(candidate_row.get(field, "")) - float(gold_row[field])) < 1e-8:
                    score_cell_matches += 1
            except (ValueError, TypeError):
                pass
    grades_score = score_cell_matches / total_score_cells if total_score_cells else 1.0

    gold_tag_pairs = {(row["student_id"], row["error_tag"]) for row in gold_tags}
    candidate_tag_pairs = {(row["student_id"], row["error_tag"]) for row in candidate_tags}
    tag_union = gold_tag_pairs | candidate_tag_pairs
    tags_score = 1.0 if not tag_union else len(gold_tag_pairs & candidate_tag_pairs) / len(tag_union)

    feedback_coverages = []
    for student_id, phrases in feedback_requirements.items():
        candidate_text = str(candidate_feedback.get(student_id, ""))
        if not phrases:
            feedback_coverages.append(1.0)
            continue
        hits = sum(1 for phrase in phrases if _lower_contains(candidate_text, phrase))
        feedback_coverages.append(hits / len(phrases))
    feedback_score = sum(feedback_coverages) / len(feedback_coverages) if feedback_coverages else 1.0

    summary_phrases = summary_requirements.get("required_phrases", [])
    summary_hits = sum(1 for phrase in summary_phrases if _lower_contains(candidate_summary, phrase))
    summary_score = summary_hits / len(summary_phrases) if summary_phrases else 1.0

    manifest_hits = sum(1 for key in REQUIRED_MANIFEST_KEYS if key in candidate_manifest)
    manifest_score = manifest_hits / len(REQUIRED_MANIFEST_KEYS)

    final_score = (grades_score + tags_score + feedback_score + summary_score + manifest_score) / 5.0
    payload.update(
        {
            "score": final_score,
            "passed": final_score == 1.0,
            "grades_score": grades_score,
            "tags_score": tags_score,
            "feedback_score": feedback_score,
            "summary_score": summary_score,
            "manifest_score": manifest_score,
            "candidate_student_ids": sorted(candidate_score_map),
            "gold_student_ids": sorted(gold_score_map),
            "candidate_tag_pairs": sorted([list(item) for item in candidate_tag_pairs]),
            "gold_tag_pairs": sorted([list(item) for item in gold_tag_pairs]),
        }
    )
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = score_submission(
        submission_dir=Path(args.submission_dir).resolve(),
        reference_dir=Path(args.reference_dir).resolve(),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
