from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

import pandas as pd


BUNDLE_ID = "educational-technology-moodle-gradebook-closeout"
COURSE_ID = "MATH-STAT-203-SP26"
COURSE_TITLE = "Applied Statistics for Public Policy"
COURSE_SHORTNAME = "MATHSTAT203-SP26"
SECTIONS = ["SEC-A", "SEC-B", "SEC-C"]
RNG_SEED = 20260418
PASS_THRESHOLD = 85

FIRST_NAMES = [
    "Alex", "Avery", "Bailey", "Blair", "Cameron", "Casey", "Devon", "Drew",
    "Elliot", "Emerson", "Finley", "Harper", "Hayden", "Jamie", "Jordan",
    "Kai", "Kendall", "Lane", "Logan", "Morgan", "Parker", "Quinn", "Reese",
    "Rowan", "Rylan", "Sawyer", "Shawn", "Skyler", "Taylor", "Wren",
]
LAST_NAMES = [
    "Adams", "Alvarez", "Bennett", "Brooks", "Campbell", "Carter", "Chen",
    "Collins", "Diaz", "Edwards", "Foster", "Garcia", "Griffin", "Hall",
    "Hughes", "Johnson", "Kim", "Lee", "Lopez", "Martinez", "Miller", "Ng",
    "Ortiz", "Patel", "Price", "Reed", "Rivera", "Singh", "Turner", "Young",
]

WITHDRAWN_IDS = {"u020", "u071", "u122", "u173", "u224"}
INCOMPLETE_IDS = {"u034", "u095", "u156", "u217"}
REENROLLED_IDS = {"u041", "u142", "u199"}
ACCOMMODATION_IDS = {"u055", "u088", "u168"}
TRANSFER_CREDIT_IDS = {"u067", "u134"}
MANUAL_OVERRIDE_IDS = {"u055", "u101", "u168", "u156"}
LOCKED_FINAL_IDS = {"u055", "u095", "u142", "u168", "u217"}

EDITABLE_BACKUP_PATHS = [
    "gradebook/policy.json",
    "gradebook/section_cutoffs.json",
    "gradebook/final_grade_flags.csv",
    "integration/id_map.csv",
]

ITEMS = [
    {
        "item_id": "quiz_01",
        "title": "Quiz 1",
        "category": "quizzes",
        "max_points": 100,
        "lateness_allowed": False,
        "oneroster_category": "quiz",
    },
    {
        "item_id": "quiz_02",
        "title": "Quiz 2",
        "category": "quizzes",
        "max_points": 100,
        "lateness_allowed": False,
        "oneroster_category": "quiz",
    },
    {
        "item_id": "quiz_03",
        "title": "Quiz 3",
        "category": "quizzes",
        "max_points": 100,
        "lateness_allowed": False,
        "oneroster_category": "quiz",
    },
    {
        "item_id": "quiz_04",
        "title": "Quiz 4",
        "category": "quizzes",
        "max_points": 100,
        "lateness_allowed": False,
        "oneroster_category": "quiz",
    },
    {
        "item_id": "hw_01",
        "title": "Homework 1",
        "category": "homework",
        "max_points": 100,
        "lateness_allowed": True,
        "oneroster_category": "homework",
    },
    {
        "item_id": "hw_02",
        "title": "Homework 2",
        "category": "homework",
        "max_points": 100,
        "lateness_allowed": True,
        "oneroster_category": "homework",
    },
    {
        "item_id": "hw_03",
        "title": "Homework 3",
        "category": "homework",
        "max_points": 100,
        "lateness_allowed": True,
        "oneroster_category": "homework",
    },
    {
        "item_id": "participation",
        "title": "Participation",
        "category": "participation",
        "max_points": 100,
        "lateness_allowed": False,
        "oneroster_category": "manual",
    },
    {
        "item_id": "midterm",
        "title": "Midterm Exam",
        "category": "midterm",
        "max_points": 100,
        "lateness_allowed": False,
        "oneroster_category": "exam",
    },
    {
        "item_id": "project",
        "title": "Policy Memo Project",
        "category": "project",
        "max_points": 100,
        "lateness_allowed": True,
        "oneroster_category": "project",
    },
    {
        "item_id": "final_exam",
        "title": "Final Exam",
        "category": "final_exam",
        "max_points": 100,
        "lateness_allowed": False,
        "oneroster_category": "exam",
    },
]

CATEGORY_ORDER = [
    "quizzes",
    "homework",
    "participation",
    "midterm",
    "project",
    "final_exam",
]

VISIBLE_CASES = [
    ("u034", "incomplete_contract"),
    ("u041", "reenrollment_id_mapping"),
    ("u055", "manual_override_and_locked_grade"),
    ("u067", "transfer_credit_substitution"),
    ("u071", "withdrawn_student_exclusion"),
    ("u088", "excused_quiz_with_drop_lowest"),
    ("u095", "incomplete_with_locked_grade"),
    ("u101", "late_penalty_vs_override"),
    ("u134", "transfer_credit_second_case"),
    ("u156", "incomplete_with_override"),
    ("u168", "accommodation_override"),
    ("u199", "reenrollment_second_case"),
]

HIDDEN_CASES = [
    ("u020", "withdrawn_hidden"),
    ("u048", "late_penalty_hidden"),
    ("u083", "missing_vs_zero_hidden"),
    ("u118", "withdrawn_hidden_second"),
    ("u142", "reenrollment_locked_hidden"),
    ("u173", "withdrawn_hidden_third"),
    ("u191", "excused_hidden"),
    ("u217", "locked_incomplete_hidden"),
]


@dataclass(frozen=True)
class BundlePaths:
    bundle_root: Path
    starter_project: Path
    reference_outputs: Path
    evaluator_only: Path
    submission: Path


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv_file(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def make_roster() -> pd.DataFrame:
    rows = []
    for index in range(1, 241):
        moodle_user_id = f"u{index:03d}"
        section_id = SECTIONS[(index - 1) // 80]
        first = FIRST_NAMES[(index - 1) % len(FIRST_NAMES)]
        last = LAST_NAMES[((index - 1) * 3) % len(LAST_NAMES)]
        current_sis = f"SIS-SP26-{index:04d}"
        legacy_sis = f"SIS-FA25-{index:04d}"
        status = "active"
        if moodle_user_id in WITHDRAWN_IDS:
            status = "withdrawn"
        elif moodle_user_id in INCOMPLETE_IDS:
            status = "incomplete"
        rows.append(
            {
                "moodle_user_id": moodle_user_id,
                "username": f"{first.lower()}.{last.lower()}{index:03d}",
                "given_name": first,
                "family_name": last,
                "section_id": section_id,
                "current_sis_student_id": current_sis,
                "legacy_sis_student_id": legacy_sis,
                "status": status,
                "reenrolled": moodle_user_id in REENROLLED_IDS,
                "accommodation_code": "EXTRA_ATTEMPT" if moodle_user_id in ACCOMMODATION_IDS else "",
                "transfer_credit": moodle_user_id in TRANSFER_CREDIT_IDS,
                "manual_override_case": moodle_user_id in MANUAL_OVERRIDE_IDS,
            }
        )
    return pd.DataFrame(rows)


def make_student_flags(roster: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for student in roster.to_dict("records"):
        moodle_user_id = student["moodle_user_id"]
        rows.append(
            {
                "moodle_user_id": moodle_user_id,
                "status": student["status"],
                "reenrolled": str(bool(student["reenrolled"])).lower(),
                "accommodation_code": student["accommodation_code"],
                "transfer_credit": str(bool(student["transfer_credit"])).lower(),
                "manual_review_reason": (
                    "incomplete_contract"
                    if student["status"] == "incomplete"
                    else ("transfer_credit_substitution" if student["transfer_credit"] else "")
                ),
            }
        )
    return pd.DataFrame(rows)


def make_item_table() -> pd.DataFrame:
    return pd.DataFrame(ITEMS)


def _base_score(student_index: int, item_index: int, section_index: int) -> float:
    return 58 + ((student_index * 17 + item_index * 13 + section_index * 7) % 36) + (
        ((student_index + item_index) % 5) * 0.8
    )


def make_submissions(roster: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    item_table = make_item_table()
    for student_pos, student in enumerate(roster.to_dict("records"), start=1):
        section_index = SECTIONS.index(student["section_id"])
        for item_pos, item in enumerate(item_table.to_dict("records"), start=1):
            raw_score = min(100.0, round(_base_score(student_pos, item_pos, section_index), 2))
            late_days = 0
            excused = False
            override_score = None
            missing = False

            if item["item_id"] in {"hw_01", "hw_02", "hw_03", "project"}:
                late_days = (student_pos + item_pos) % 4

            if student["moodle_user_id"] == "u034" and item["item_id"] == "quiz_02":
                excused = True
                raw_score = None
            if student["moodle_user_id"] == "u088" and item["item_id"] == "quiz_01":
                excused = True
                raw_score = None
            if student["moodle_user_id"] == "u191" and item["item_id"] == "quiz_03":
                excused = True
                raw_score = None
            if student["moodle_user_id"] == "u156" and item["item_id"] == "quiz_04":
                excused = True
                raw_score = None

            if student["moodle_user_id"] in {"u048", "u101", "u168"} and item["item_id"] in {"hw_02", "project"}:
                late_days = 3

            if student["moodle_user_id"] in {"u067", "u134"} and item["item_id"] == "hw_03":
                override_score = 91.0 if student["moodle_user_id"] == "u067" else 88.0
                raw_score = None
                missing = True

            if student["moodle_user_id"] in {"u055", "u101", "u156", "u168"} and item["item_id"] in {"project", "midterm"}:
                override_score = min(100.0, raw_score + 12.0)

            if student["moodle_user_id"] in {"u083", "u156"} and item["item_id"] == "hw_02":
                raw_score = None
                missing = True

            if item["item_id"] == "participation":
                raw_score = float(100 - ((student_pos + section_index) % 5) * 5)
                late_days = 0

            rows.append(
                {
                    "moodle_user_id": student["moodle_user_id"],
                    "item_id": item["item_id"],
                    "raw_score": "" if raw_score is None else round(float(raw_score), 2),
                    "late_days": late_days,
                    "excused": str(bool(excused)).lower(),
                    "override_score": "" if override_score is None else round(float(override_score), 2),
                    "missing": str(bool(missing or raw_score is None)).lower(),
                }
            )
    return pd.DataFrame(rows)


def canonical_policy() -> dict:
    return {
        "category_weights": {
            "quizzes": 20,
            "homework": 20,
            "participation": 5,
            "midterm": 20,
            "project": 15,
            "final_exam": 20,
        },
        "drop_lowest": {"quizzes": 1},
        "empty_grade_behavior": "zero",
        "excused_behavior": "exclude_from_denominator",
        "override_precedence": "manual_override_first",
        "late_policy": {
            "enabled": True,
            "per_day_fraction": 0.10,
            "max_fraction": 0.30,
            "double_apply": False,
        },
    }


def broken_policy() -> dict:
    return {
        "category_weights": {
            "quizzes": 15,
            "homework": 25,
            "participation": 5,
            "midterm": 20,
            "project": 15,
            "final_exam": 20,
        },
        "drop_lowest": {"quizzes": 0},
        "empty_grade_behavior": "exclude",
        "excused_behavior": "count_as_zero",
        "override_precedence": "raw_score_first",
        "late_policy": {
            "enabled": True,
            "per_day_fraction": 0.10,
            "max_fraction": 0.30,
            "double_apply": True,
        },
    }


def canonical_cutoffs() -> dict:
    standard = [
        ["A", 93],
        ["A-", 90],
        ["B+", 87],
        ["B", 83],
        ["B-", 80],
        ["C+", 77],
        ["C", 73],
        ["C-", 70],
        ["D", 60],
        ["F", 0],
    ]
    stricter = [
        ["A", 94],
        ["A-", 91],
        ["B+", 88],
        ["B", 84],
        ["B-", 81],
        ["C+", 78],
        ["C", 74],
        ["C-", 70],
        ["D", 60],
        ["F", 0],
    ]
    return {
        "SEC-A": standard,
        "SEC-B": standard,
        "SEC-C": stricter,
    }


def broken_cutoffs() -> dict:
    standard = canonical_cutoffs()["SEC-A"]
    return {section: standard for section in SECTIONS}


def canonical_id_map(roster: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for student in roster.to_dict("records"):
        rows.append(
            {
                "moodle_user_id": student["moodle_user_id"],
                "sis_student_id": student["current_sis_student_id"],
                "legacy_sis_student_id": student["legacy_sis_student_id"],
            }
        )
    return pd.DataFrame(rows)


def broken_id_map(roster: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for student in roster.to_dict("records"):
        sis_id = student["current_sis_student_id"]
        if student["moodle_user_id"] in REENROLLED_IDS:
            sis_id = student["legacy_sis_student_id"]
        rows.append(
            {
                "moodle_user_id": student["moodle_user_id"],
                "sis_student_id": sis_id,
                "legacy_sis_student_id": student["legacy_sis_student_id"],
            }
        )
    return pd.DataFrame(rows)


def canonical_final_grade_flags(roster: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "moodle_user_id": student["moodle_user_id"],
                "locked": "false",
                "cached_numeric_grade": "",
                "cached_letter_grade": "",
            }
            for student in roster.to_dict("records")
        ]
    )


def broken_final_grade_flags(roster: pd.DataFrame, expected_export: pd.DataFrame) -> pd.DataFrame:
    expected_lookup = expected_export.set_index("moodle_user_id").to_dict("index")
    rows = []
    for student in roster.to_dict("records"):
        moodle_user_id = student["moodle_user_id"]
        locked = moodle_user_id in LOCKED_FINAL_IDS
        cached_numeric = ""
        cached_letter = ""
        if locked and moodle_user_id in expected_lookup:
            numeric = expected_lookup[moodle_user_id]["final_numeric_grade"]
            stale_numeric = round(max(0.0, float(numeric) - 6.5), 2)
            cached_numeric = f"{stale_numeric:.2f}"
            cached_letter = "B-" if stale_numeric >= 80 else "C"
        rows.append(
            {
                "moodle_user_id": moodle_user_id,
                "locked": str(bool(locked)).lower(),
                "cached_numeric_grade": cached_numeric,
                "cached_letter_grade": cached_letter,
            }
        )
    return pd.DataFrame(rows)


def _score_to_float(value: object) -> float | None:
    if value in ("", None):
        return None
    return float(value)


def _drop_lowest(series: list[tuple[str, float]], count: int) -> set[str]:
    ordered = sorted(series, key=lambda item: (item[1], item[0]))
    return {item_id for item_id, _ in ordered[:count]}


def letter_from_score(score: float, section_id: str, cutoffs: dict) -> str:
    for label, threshold in cutoffs[section_id]:
        if score >= float(threshold):
            return label
    return "F"


def compute_grade_outputs(
    roster: pd.DataFrame,
    items: pd.DataFrame,
    submissions: pd.DataFrame,
    policy: dict,
    section_cutoffs: dict,
    id_map: pd.DataFrame,
    final_grade_flags: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    roster_lookup = roster.set_index("moodle_user_id").to_dict("index")
    item_lookup = items.set_index("item_id").to_dict("index")
    id_lookup = id_map.set_index("moodle_user_id").to_dict("index")
    flags_lookup = final_grade_flags.set_index("moodle_user_id").to_dict("index")

    detail_rows: list[dict] = []
    audit_rows: list[dict] = []
    export_rows: list[dict] = []

    for moodle_user_id, student_rows in submissions.groupby("moodle_user_id"):
        student = roster_lookup[moodle_user_id]
        rows = student_rows.to_dict("records")
        category_scores: dict[str, list[tuple[str, float]]] = {category: [] for category in CATEGORY_ORDER}
        late_penalty_points = 0.0
        override_count = 0
        excused_count = 0

        for row in rows:
            item = item_lookup[row["item_id"]]
            raw_score = _score_to_float(row["raw_score"])
            override_score = _score_to_float(row["override_score"])
            missing = row["missing"] == "true"
            excused = row["excused"] == "true"
            late_days = int(row["late_days"])

            if excused:
                excused_count += 1
                if policy["excused_behavior"] == "exclude_from_denominator":
                    effective_score = None
                    status = "excused"
                else:
                    effective_score = 0.0
                    status = "excused_counted_zero"
            else:
                source_score = raw_score
                if override_score is not None:
                    override_count += 1
                    if policy["override_precedence"] == "manual_override_first":
                        source_score = override_score
                if source_score is None:
                    if policy["empty_grade_behavior"] == "exclude":
                        effective_score = None
                        status = "excluded_empty"
                    else:
                        effective_score = 0.0
                        status = "missing_zero"
                else:
                    effective_score = float(source_score)
                    status = "graded"

                if (
                    effective_score is not None
                    and late_days > 0
                    and item["lateness_allowed"]
                    and policy["late_policy"]["enabled"]
                ):
                    penalty_fraction = min(
                        policy["late_policy"]["max_fraction"],
                        policy["late_policy"]["per_day_fraction"] * late_days,
                    )
                    penalty_applied = penalty_fraction
                    if policy["late_policy"]["double_apply"]:
                        penalty_applied = min(
                            policy["late_policy"]["max_fraction"],
                            penalty_applied * 2,
                        )
                    penalized = round(effective_score * (1 - penalty_applied), 2)
                    late_penalty_points += max(0.0, effective_score - penalized)
                    effective_score = penalized
                    status = f"{status}_late"

            percent = None if effective_score is None else round((effective_score / item["max_points"]) * 100, 4)
            if percent is not None:
                category_scores[item["category"]].append((row["item_id"], percent))

            detail_rows.append(
                {
                    "moodle_user_id": moodle_user_id,
                    "item_id": row["item_id"],
                    "status": status,
                    "effective_score": "" if effective_score is None else round(effective_score, 2),
                    "effective_percent": "" if percent is None else round(percent, 4),
                }
            )

        dropped_items: set[str] = set()
        dropped_count = int(policy["drop_lowest"].get("quizzes", 0))
        if dropped_count:
            dropped_items = _drop_lowest(category_scores["quizzes"], dropped_count)

        category_percents: dict[str, float | None] = {}
        for category in CATEGORY_ORDER:
            scores = category_scores[category]
            filtered = [
                percent
                for item_id, percent in scores
                if item_id not in dropped_items
            ]
            category_percents[category] = round(sum(filtered) / len(filtered), 4) if filtered else None

        weighted_sum = 0.0
        for category, weight in policy["category_weights"].items():
            category_percent = category_percents.get(category)
            if category_percent is None:
                category_percent = 0.0
            weighted_sum += category_percent * (float(weight) / 100.0)

        final_numeric = round(weighted_sum, 2)
        export_status = "posted"
        final_letter = letter_from_score(final_numeric, student["section_id"], section_cutoffs)
        if student["status"] == "incomplete":
            export_status = "incomplete"
            final_letter = "I"
        elif student["status"] == "withdrawn":
            export_status = "withdrawn"
            final_letter = "W"

        flags = flags_lookup[moodle_user_id]
        exported_numeric = final_numeric
        exported_letter = final_letter
        if flags["locked"] == "true" and export_status != "withdrawn":
            if flags["cached_numeric_grade"] not in ("", None):
                exported_numeric = round(float(flags["cached_numeric_grade"]), 2)
            if flags["cached_letter_grade"] not in ("", None):
                exported_letter = str(flags["cached_letter_grade"])

        audit_rows.append(
            {
                "moodle_user_id": moodle_user_id,
                "sis_student_id": id_lookup[moodle_user_id]["sis_student_id"],
                "section_id": student["section_id"],
                "status": export_status,
                "quiz_percent": "" if category_percents["quizzes"] is None else round(category_percents["quizzes"], 2),
                "homework_percent": "" if category_percents["homework"] is None else round(category_percents["homework"], 2),
                "participation_percent": "" if category_percents["participation"] is None else round(category_percents["participation"], 2),
                "midterm_percent": "" if category_percents["midterm"] is None else round(category_percents["midterm"], 2),
                "project_percent": "" if category_percents["project"] is None else round(category_percents["project"], 2),
                "final_exam_percent": "" if category_percents["final_exam"] is None else round(category_percents["final_exam"], 2),
                "dropped_item_ids": ";".join(sorted(dropped_items)),
                "late_penalty_points": round(late_penalty_points, 2),
                "override_count": override_count,
                "excused_count": excused_count,
                "final_numeric_grade": round(final_numeric, 2),
                "final_letter_grade": final_letter,
                "exported_numeric_grade": round(exported_numeric, 2),
                "exported_letter_grade": exported_letter,
            }
        )

        if export_status != "withdrawn":
            export_rows.append(
                {
                    "moodle_user_id": moodle_user_id,
                    "sis_student_id": id_lookup[moodle_user_id]["sis_student_id"],
                    "section_id": student["section_id"],
                    "final_numeric_grade": f"{exported_numeric:.2f}",
                    "final_letter_grade": exported_letter,
                    "export_status": export_status,
                }
            )

    return pd.DataFrame(detail_rows), pd.DataFrame(audit_rows), pd.DataFrame(export_rows)


def build_exception_log(audit_rows: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    roster_lookup = roster.set_index("moodle_user_id").to_dict("index")
    rows = []
    for audit_row in audit_rows.to_dict("records"):
        student = roster_lookup[audit_row["moodle_user_id"]]
        reason = ""
        if student["status"] == "incomplete":
            reason = "incomplete_contract"
        elif student["transfer_credit"]:
            reason = "transfer_credit_substitution"
        if reason:
            rows.append(
                {
                    "sis_student_id": audit_row["sis_student_id"],
                    "section_id": audit_row["section_id"],
                    "reason_code": reason,
                    "note": "manual review retained by institutional policy",
                }
            )
    return pd.DataFrame(rows)


def build_registrar_export_xml(final_export: pd.DataFrame) -> str:
    root = ET.Element("registrarGradeExport", attrib={"courseId": COURSE_ID})
    for row in final_export.sort_values(["section_id", "sis_student_id"]).to_dict("records"):
        node = ET.SubElement(root, "student")
        ET.SubElement(node, "sisStudentId").text = row["sis_student_id"]
        ET.SubElement(node, "sectionId").text = row["section_id"]
        ET.SubElement(node, "finalNumericGrade").text = row["final_numeric_grade"]
        ET.SubElement(node, "finalLetterGrade").text = row["final_letter_grade"]
        ET.SubElement(node, "status").text = row["export_status"]
    return ET.tostring(root, encoding="unicode")


def build_oneroster_tables(
    roster: pd.DataFrame,
    items: pd.DataFrame,
    detail_rows: pd.DataFrame,
    final_export: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    users = roster[["current_sis_student_id", "username", "given_name", "family_name", "status"]].copy()
    users.columns = ["sourcedId", "username", "givenName", "familyName", "status"]

    classes = pd.DataFrame(
        [
            {
                "sourcedId": section_id,
                "title": f"{COURSE_TITLE} {section_id}",
                "courseSourcedId": COURSE_ID,
                "classCode": section_id,
            }
            for section_id in SECTIONS
        ]
    )

    enrollments = roster[["moodle_user_id", "section_id", "current_sis_student_id", "status"]].copy()
    enrollments["sourcedId"] = enrollments["moodle_user_id"] + "-" + enrollments["section_id"]
    enrollments["role"] = "student"
    enrollments["primary"] = "true"
    enrollments = enrollments[
        ["sourcedId", "section_id", "current_sis_student_id", "role", "primary", "status"]
    ]
    enrollments.columns = [
        "sourcedId",
        "classSourcedId",
        "userSourcedId",
        "role",
        "primary",
        "status",
    ]

    line_items = []
    for item in items.to_dict("records"):
        line_items.append(
            {
                "sourcedId": item["item_id"],
                "classSourcedId": COURSE_ID,
                "title": item["title"],
                "category": item["oneroster_category"],
                "maxScore": item["max_points"],
            }
        )
    line_items.append(
        {
            "sourcedId": "final_grade",
            "classSourcedId": COURSE_ID,
            "title": "Final Grade",
            "category": "final",
            "maxScore": 100,
        }
    )
    line_items_df = pd.DataFrame(line_items)

    detail_lookup = detail_rows.set_index(["moodle_user_id", "item_id"]).to_dict("index")
    roster_lookup = roster.set_index("moodle_user_id").to_dict("index")
    results_rows: list[dict] = []
    for moodle_user_id, student in roster_lookup.items():
        if student["status"] == "withdrawn":
            continue
        for item in items.to_dict("records"):
            detail = detail_lookup[(moodle_user_id, item["item_id"])]
            score = detail["effective_score"]
            status = "exempt" if detail["status"].startswith("excused") else ("missing" if detail["status"] == "missing_zero" else "completed")
            results_rows.append(
                {
                    "sourcedId": f"{moodle_user_id}-{item['item_id']}",
                    "lineItemSourcedId": item["item_id"],
                    "studentSourcedId": student["current_sis_student_id"],
                    "score": "" if score == "" else f"{float(score):.2f}",
                    "status": status,
                }
            )

        final_row = final_export.set_index("moodle_user_id").to_dict("index")[moodle_user_id]
        results_rows.append(
            {
                "sourcedId": f"{moodle_user_id}-final_grade",
                "lineItemSourcedId": "final_grade",
                "studentSourcedId": student["current_sis_student_id"],
                "score": final_row["final_numeric_grade"],
                "status": final_row["export_status"],
            }
        )

    results_df = pd.DataFrame(results_rows)
    return {
        "users.csv": users,
        "classes.csv": classes,
        "enrollments.csv": enrollments,
        "lineItems.csv": line_items_df,
        "results.csv": results_df,
    }


def build_manifest(rows_by_file: dict[str, bytes]) -> pd.DataFrame:
    rows = []
    for filename, content in sorted(rows_by_file.items()):
        row_count = max(0, content.decode("utf-8").count("\n") - 1) if filename.endswith(".csv") else ""
        rows.append(
            {
                "fileName": filename,
                "rowCount": row_count,
                "sha256": sha256_bytes(content),
            }
        )
    return pd.DataFrame(rows)


def simple_pdf_from_lines(lines: list[str]) -> bytes:
    escaped = []
    for index, line in enumerate(lines):
        line = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        if index == 0:
            escaped.append(f"BT /F1 10 Tf 50 780 Td ({line}) Tj")
        else:
            escaped.append(f"0 -14 Td ({line}) Tj")
    escaped.append("ET")
    stream = "\n".join(escaped).encode("utf-8")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 5 0 R /Resources << /Font << /F1 4 0 R >> >> >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(stream)} >>\nstream\n".encode("utf-8") + stream + b"\nendstream",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode("utf-8"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")
    xref_start = len(pdf)
    pdf.extend(f"xref\n0 {len(offsets)}\n".encode("utf-8"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("utf-8"))
    pdf.extend(
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF\n".encode(
            "utf-8"
        )
    )
    return bytes(pdf)


def write_policy_pdf(path: Path, canonical: bool) -> None:
    policy = canonical_policy()
    cutoffs = canonical_cutoffs()
    lines = [
        "Applied Statistics for Public Policy - Spring 2026 grading policy",
        "",
        f"Category weights: quizzes={policy['category_weights']['quizzes']}%, homework={policy['category_weights']['homework']}%, participation={policy['category_weights']['participation']}%, midterm={policy['category_weights']['midterm']}%, project={policy['category_weights']['project']}%, final={policy['category_weights']['final_exam']}%",
        "Drop the single lowest quiz score after excused quizzes are excluded.",
        "Missing work counts as zero. Excused work is excluded from the denominator.",
        "Late penalty for homework/project: 10% per calendar day, maximum 30%, applied once.",
        "Manual override grades supersede raw grades.",
        "Incomplete contracts remain exportable with status INCOMPLETE and letter grade I.",
        f"SEC-A/SEC-B cutoffs: {cutoffs['SEC-A']}",
        f"SEC-C cutoffs: {cutoffs['SEC-C']}",
    ]
    if not canonical:
        lines.append("This document is authoritative; the starter backup does not currently match it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(simple_pdf_from_lines(lines))


def build_moodle_backup_xml() -> str:
    root = ET.Element("moodle_backup")
    info = ET.SubElement(root, "information")
    ET.SubElement(info, "courseid").text = COURSE_ID
    ET.SubElement(info, "fullname").text = COURSE_TITLE
    ET.SubElement(info, "shortname").text = COURSE_SHORTNAME
    ET.SubElement(info, "format").text = "benchmark_offline_moodle2"
    ET.SubElement(info, "sections").text = str(len(SECTIONS))
    ET.SubElement(info, "activities").text = str(len(ITEMS))
    return ET.tostring(root, encoding="unicode")


def df_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def build_backup_archive(
    backup_path: Path,
    roster: pd.DataFrame,
    items: pd.DataFrame,
    submissions: pd.DataFrame,
    policy: dict,
    section_cutoffs: dict,
    id_map: pd.DataFrame,
    final_grade_flags: pd.DataFrame,
) -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        temp = Path(tempdir)
        files_to_write: dict[str, bytes] = {
            "moodle_backup.xml": build_moodle_backup_xml().encode("utf-8"),
            "course/course.json": json.dumps(
                {
                    "course_id": COURSE_ID,
                    "title": COURSE_TITLE,
                    "shortname": COURSE_SHORTNAME,
                    "section_count": len(SECTIONS),
                    "graded_item_count": len(ITEMS),
                },
                indent=2,
                sort_keys=True,
            ).encode("utf-8"),
            "course/sections.csv": pd.DataFrame(
                [
                    {"section_id": section_id, "title": f"Section {section_id[-1]}", "teacher_count": 2}
                    for section_id in SECTIONS
                ]
            ).to_csv(index=False).encode("utf-8"),
            "gradebook/items.csv": df_to_csv_bytes(items),
            "gradebook/submissions.csv": df_to_csv_bytes(submissions),
            "gradebook/student_flags.csv": df_to_csv_bytes(make_student_flags(roster)),
            "gradebook/policy.json": json.dumps(policy, indent=2, sort_keys=True).encode("utf-8"),
            "gradebook/section_cutoffs.json": json.dumps(section_cutoffs, indent=2, sort_keys=True).encode("utf-8"),
            "gradebook/final_grade_flags.csv": df_to_csv_bytes(final_grade_flags),
            "integration/id_map.csv": df_to_csv_bytes(id_map),
            "README.txt": (
                "This benchmark .mbz is a deterministic Moodle-style backup artifact used by the "
                "Agent-HLE educational-technology task. It is intended for offline repair and export "
                "reconstruction rather than direct Moodle server restore.\n"
            ).encode("utf-8"),
        }

        immutable_hashes = {}
        for relative_path, content in files_to_write.items():
            path = temp / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            if relative_path not in EDITABLE_BACKUP_PATHS and relative_path != "benchmark_contract.json":
                immutable_hashes[relative_path] = sha256_bytes(content)

        contract = {
            "editable_paths": EDITABLE_BACKUP_PATHS,
            "immutable_hashes": immutable_hashes,
            "bundle_id": BUNDLE_ID,
        }
        (temp / "benchmark_contract.json").write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        backup_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(backup_path, "w", compression=zipfile.ZIP_DEFLATED) as handle:
            for path in sorted(temp.rglob("*")):
                if path.is_file():
                    handle.write(path, arcname=str(path.relative_to(temp)))


def extract_backup_archive(backup_path: Path, output_dir: Path) -> None:
    ensure_clean_dir(output_dir)
    with zipfile.ZipFile(backup_path) as handle:
        handle.extractall(output_dir)


def read_backup_state(backup_path: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tempdir:
        temp = Path(tempdir)
        extract_backup_archive(backup_path, temp)
        state = {
            "course": json.loads((temp / "course" / "course.json").read_text()),
            "policy": json.loads((temp / "gradebook" / "policy.json").read_text()),
            "section_cutoffs": json.loads((temp / "gradebook" / "section_cutoffs.json").read_text()),
            "items": pd.read_csv(temp / "gradebook" / "items.csv"),
            "submissions": pd.read_csv(temp / "gradebook" / "submissions.csv", keep_default_na=False),
            "student_flags": pd.read_csv(temp / "gradebook" / "student_flags.csv", keep_default_na=False),
            "id_map": pd.read_csv(temp / "integration" / "id_map.csv"),
            "final_grade_flags": pd.read_csv(temp / "gradebook" / "final_grade_flags.csv", keep_default_na=False),
            "contract": json.loads((temp / "benchmark_contract.json").read_text()),
            "file_hashes": {
                str(path.relative_to(temp)): sha256_path(path)
                for path in temp.rglob("*")
                if path.is_file()
            },
        }
        return state


def build_submission_outputs(
    bundle_root: Path,
    backup_path: Path,
    roster_path: Path,
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    state = read_backup_state(backup_path)
    roster = pd.read_csv(roster_path)
    detail_rows, audit_rows, final_export = compute_grade_outputs(
        roster=roster,
        items=state["items"],
        submissions=state["submissions"],
        policy=state["policy"],
        section_cutoffs=state["section_cutoffs"],
        id_map=state["id_map"],
        final_grade_flags=state["final_grade_flags"],
    )
    exception_log = build_exception_log(audit_rows, roster)

    final_export_csv = final_export[
        ["sis_student_id", "section_id", "final_numeric_grade", "final_letter_grade", "export_status"]
    ].sort_values(["section_id", "sis_student_id"])
    final_export_csv.to_csv(output_dir / "final_grade_export.csv", index=False)
    write_text(output_dir / "final_grade_export.xml", build_registrar_export_xml(final_export_csv))

    oneroster_tables = build_oneroster_tables(roster, state["items"], detail_rows, final_export)
    oneroster_dir = output_dir / "oneroster_package"
    oneroster_dir.mkdir(parents=True, exist_ok=True)
    manifest_sources: dict[str, bytes] = {}
    for filename, frame in oneroster_tables.items():
        csv_bytes = df_to_csv_bytes(frame)
        (oneroster_dir / filename).write_bytes(csv_bytes)
        manifest_sources[filename] = csv_bytes
    manifest = build_manifest(manifest_sources)
    manifest.to_csv(oneroster_dir / "manifest.csv", index=False)

    audit_rows.sort_values(["section_id", "sis_student_id"]).to_csv(output_dir / "audit_report.csv", index=False)
    audit_summary = {
        "course_id": COURSE_ID,
        "exported_students": int(len(final_export_csv)),
        "visible_case_ids": [student_id for student_id, _ in VISIBLE_CASES],
        "hidden_case_ids": [student_id for student_id, _ in HIDDEN_CASES],
        "final_grade_sha256": sha256_path(output_dir / "final_grade_export.csv"),
    }
    write_json(output_dir / "audit_report.json", audit_summary)
    exception_log.sort_values(["section_id", "sis_student_id"]).to_csv(output_dir / "exception_log.csv", index=False)

    decisions_text = (
        "# Decisions\n\n"
        "- repaired category weights to the official policy\n"
        "- restored drop-lowest quiz behavior\n"
        "- restored manual override precedence\n"
        "- fixed late-penalty application to a single capped pass\n"
        "- removed stale locked-grade exports\n"
        "- restored excused-vs-missing semantics\n"
        "- restored section-specific cutoffs for SEC-C\n"
        "- repaired SIS identifier mapping for re-enrolled students\n"
    )
    write_text(output_dir / "decisions.md", decisions_text)
    shutil.copy2(backup_path, output_dir / "corrected_course.mbz")

    return {
        "detail_rows": detail_rows,
        "audit_rows": audit_rows,
        "final_export": final_export_csv,
        "exception_log": exception_log,
    }


def make_policy_markdown() -> str:
    return (
        "# Registrar Closeout Rules\n\n"
        "Use the policy PDF as the source of truth. The key operational rules are:\n\n"
        "1. Quizzes are weighted at 20% and the single lowest non-excused quiz is dropped.\n"
        "2. Homework is weighted at 20%.\n"
        "3. Participation is weighted at 5%.\n"
        "4. Midterm is weighted at 20%.\n"
        "5. Project is weighted at 15%.\n"
        "6. Final exam is weighted at 20%.\n"
        "7. Missing work counts as zero. Excused work is excluded from the denominator.\n"
        "8. Homework and project late penalties are 10% per day, capped at 30%, applied once.\n"
        "9. Manual overrides supersede raw scores.\n"
        "10. Incomplete contracts remain exportable with status `incomplete` and letter grade `I`.\n"
    )


def write_starter_helper_scripts(bundle_root: Path, starter_tools_dir: Path) -> None:
    helper = (
        "from __future__ import annotations\n\n"
        "import argparse\n"
        "import sys\n"
        "from pathlib import Path\n\n"
        "ROOT = Path(__file__).resolve().parents[2]\n"
        "if str(ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(ROOT))\n\n"
        "from bundle_lib import build_submission_outputs, extract_backup_archive\n\n"
        "def parse_args() -> argparse.Namespace:\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('--backup', type=Path, required=True)\n"
        "    parser.add_argument('--roster', type=Path, required=False)\n"
        "    parser.add_argument('--output', type=Path, required=True)\n"
        "    return parser.parse_args()\n\n"
        "def main() -> int:\n"
        "    args = parse_args()\n"
        "    roster = args.roster or (ROOT / 'starter_project' / 'roster.csv')\n"
        "    build_submission_outputs(ROOT, args.backup, roster, args.output)\n"
        "    return 0\n\n"
        "if __name__ == '__main__':\n"
        "    raise SystemExit(main())\n"
    )
    extract = (
        "from __future__ import annotations\n\n"
        "import argparse\n"
        "import sys\n"
        "from pathlib import Path\n\n"
        "ROOT = Path(__file__).resolve().parents[2]\n"
        "if str(ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(ROOT))\n\n"
        "from bundle_lib import extract_backup_archive\n\n"
        "def parse_args() -> argparse.Namespace:\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('--backup', type=Path, required=True)\n"
        "    parser.add_argument('--output', type=Path, required=True)\n"
        "    return parser.parse_args()\n\n"
        "def main() -> int:\n"
        "    args = parse_args()\n"
        "    extract_backup_archive(args.backup, args.output)\n"
        "    return 0\n\n"
        "if __name__ == '__main__':\n"
        "    raise SystemExit(main())\n"
    )
    write_text(starter_tools_dir / "rebuild_exports.py", helper)
    write_text(starter_tools_dir / "extract_backup.py", extract)
    write_text(
        starter_tools_dir / "README.md",
        "# Starter tools\n\n"
        "- `extract_backup.py` unzips the `.mbz` archive into a working directory.\n"
        "- `rebuild_exports.py` rebuilds the registrar and OneRoster exports from a corrected backup.\n",
    )


def write_bundle_docs(paths: BundlePaths) -> None:
    write_text(
        paths.starter_project / "env_setup.md",
        "Use the staged `software/python_with_task_deps.sh` wrapper for the pinned Python + pandas runtime. No live Moodle server or external database is required for the benchmark workflow.\n",
    )
    write_text(paths.starter_project / "GRADING_POLICY.md", make_policy_markdown())
    write_text(
        paths.starter_project / "README.md",
        "Starter bundle for the Moodle gradebook closeout task. Repair the backup, then rebuild exports.\n",
    )


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def package_zip_from_dir(src_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(src_dir.rglob("*")):
            if path.is_file():
                handle.write(path, arcname=str(path.relative_to(src_dir)))


def default_paths(bundle_root: Path) -> BundlePaths:
    return BundlePaths(
        bundle_root=bundle_root,
        starter_project=bundle_root / "starter_project",
        reference_outputs=bundle_root / "reference_outputs",
        evaluator_only=bundle_root / "evaluator_only",
        submission=bundle_root / "submission",
    )
