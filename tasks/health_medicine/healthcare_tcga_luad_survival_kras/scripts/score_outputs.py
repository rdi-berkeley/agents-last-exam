#!/usr/bin/env python
"""Weighted evaluator for the TCGA-LUAD KRAS survival task.

The public GDC catalogue is mutable, so the frozen reference is used as a
high-coverage verification oracle rather than as an exact answer key.  The
submitted cohort is also used to independently recompute the median split,
log-rank test, and Efron-ties Cox model before the reported JSON is scored.
"""

from __future__ import annotations

import binascii
import csv
import io
import json
import math
import re
import shutil
import statistics
import struct
import subprocess
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EVALUATOR_VERSION = "2.0.0"

REQUIRED_FILES = [
    "cohort.csv",
    "km_plot.png",
    "cox_results.json",
    "analysis.R",
]

REQUIRED_COHORT_COLUMNS = [
    "patient_id",
    "sample_id",
    "file_id",
    "kras_expression_selected",
    "kras_group",
    "time_days",
    "status",
    "age_at_diagnosis_years",
    "raw_stage",
    "stage_group",
]

SECTION_WEIGHTS = {
    "artifacts": 0.20,
    "cohort": 0.25,
    "derivations": 0.25,
    "statistics": 0.20,
    "reproducibility": 0.10,
}

EXPECTED_COX_VARIABLES = (
    "kras_group_high",
    "age_at_diagnosis_years",
    "stage_group_advanced",
)


@dataclass
class ScoreResult:
    score: float
    passed: bool
    reasons: list[str]
    details: dict[str, object]
    sections: dict[str, float]
    warnings: list[str]
    hard_failure: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "evaluator_version": EVALUATOR_VERSION,
            "score": self.score,
            "passed": self.passed,
            "hard_failure": self.hard_failure,
            "sections": self.sections,
            "reasons": self.reasons,
            "warnings": self.warnings,
            "details": self.details,
        }


def _hard_failure(
    reason: str, *, details: dict[str, object] | None = None
) -> ScoreResult:
    return ScoreResult(
        score=0.0,
        passed=False,
        reasons=[reason],
        details=details or {},
        sections={name: 0.0 for name in SECTION_WEIGHTS},
        warnings=[],
        hard_failure=True,
    )


def _load_csv(payload: bytes, label: str) -> tuple[list[str], list[dict[str, str]]]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not valid UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text), strict=True)
    fieldnames = reader.fieldnames or []
    if len(fieldnames) != len(set(fieldnames)):
        raise ValueError(f"{label} contains duplicate column names")
    try:
        rows = list(reader)
    except csv.Error as exc:
        raise ValueError(f"{label} is not valid CSV: {exc}") from exc
    if any(None in row for row in rows):
        raise ValueError(f"{label} contains rows with too many fields")
    return fieldnames, rows


def _load_json(payload: bytes, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value}")

    try:
        data = json.loads(payload.decode("utf-8"), parse_constant=reject_constant)
    except Exception as exc:
        raise ValueError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return data


def _as_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_int(value: object) -> int | None:
    number = _as_float(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _band_score(value: float, *, zero_below: float, full_at: float) -> float:
    if value >= full_at:
        return 1.0
    if value <= zero_below:
        return 0.0
    return (value - zero_below) / (full_at - zero_below)


def _close_score(
    observed: float | None,
    expected: float | None,
    *,
    atol: float,
    rtol: float = 0.0,
) -> float:
    if observed is None or expected is None:
        return 0.0
    tolerance = atol + rtol * abs(expected)
    difference = abs(observed - expected)
    if difference <= tolerance:
        return 1.0
    if tolerance == 0.0 or difference >= 5.0 * tolerance:
        return 0.0
    return 1.0 - (difference - tolerance) / (4.0 * tolerance)


def _normalize_stage(raw_stage: object) -> str | None:
    value = str(raw_stage or "").strip().upper()
    value = re.sub(r"^STAGE\s+", "", value)
    value = re.sub(r"\s+", "", value)
    value = value.replace("[", "").replace("]", "")
    early = {"I", "IA", "IA1", "IA2", "IA3", "IB", "II", "IIA", "IIB"}
    advanced = {"III", "IIIA", "IIIB", "IIIC", "IV", "IVA", "IVB"}
    if value in early:
        return "early"
    if value in advanced:
        return "advanced"
    return None


def _parse_cohort(
    payload: bytes,
    *,
    minimum_patient_count: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    fieldnames, raw_rows = _load_csv(payload, "cohort.csv")
    missing_columns = [
        name for name in REQUIRED_COHORT_COLUMNS if name not in fieldnames
    ]
    if missing_columns:
        raise ValueError(f"cohort.csv missing columns: {missing_columns}")
    if len(raw_rows) < minimum_patient_count:
        raise ValueError(
            f"cohort.csv has too few patients: {len(raw_rows)} < {minimum_patient_count}"
        )

    patient_ids = [str(row.get("patient_id") or "").strip() for row in raw_rows]
    duplicates = sorted(
        {patient_id for patient_id in patient_ids if patient_ids.count(patient_id) > 1}
    )
    if duplicates:
        preview = duplicates[:5]
        raise ValueError(f"cohort.csv contains duplicate patient_id values: {preview}")

    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_rows, start=2):
        patient_id = str(raw.get("patient_id") or "").strip()
        sample_id = str(raw.get("sample_id") or "").strip()
        file_id = str(raw.get("file_id") or "").strip()
        group = str(raw.get("kras_group") or "").strip().lower()
        stage_group = str(raw.get("stage_group") or "").strip().lower()
        expression = _as_float(raw.get("kras_expression_selected"))
        time_days = _as_float(raw.get("time_days"))
        status = _as_int(raw.get("status"))
        age = _as_float(raw.get("age_at_diagnosis_years"))
        if not patient_id or not sample_id or not file_id:
            raise ValueError(f"cohort.csv row {index} has an empty identifier")
        if expression is None or expression < 0:
            raise ValueError(f"cohort.csv row {index} has invalid KRAS expression")
        if time_days is None or time_days < 0:
            raise ValueError(f"cohort.csv row {index} has invalid time_days")
        if status not in {0, 1}:
            raise ValueError(f"cohort.csv row {index} has invalid status")
        if age is None or age <= 0 or age > 130:
            raise ValueError(f"cohort.csv row {index} has invalid age")
        if group not in {"high", "low"}:
            raise ValueError(f"cohort.csv row {index} has invalid kras_group")
        if stage_group not in {"early", "advanced"}:
            raise ValueError(f"cohort.csv row {index} has invalid stage_group")
        rows.append(
            {
                "patient_id": patient_id,
                "sample_id": sample_id[:15],
                "file_id": file_id,
                "expression": expression,
                "group": group,
                "time_days": time_days,
                "status": status,
                "age": age,
                "raw_stage": str(raw.get("raw_stage") or "").strip(),
                "stage_group": stage_group,
            }
        )
    return rows, fieldnames


def _reference_rows(reference_cohort_csv: bytes) -> dict[str, dict[str, Any]]:
    _, rows = _load_csv(reference_cohort_csv, "reference cohort.csv")
    result: dict[str, dict[str, Any]] = {}
    for raw in rows:
        patient_id = str(raw.get("patient_id") or "").strip()
        if not patient_id:
            continue
        result[patient_id] = {
            "patient_id": patient_id,
            "sample_id": str(raw.get("sample_id") or "").strip()[:15],
            "file_id": str(raw.get("file_id") or "").strip(),
            "expression": _as_float(raw.get("kras_expression_selected")),
            "time_days": _as_float(raw.get("time_days")),
            "status": _as_int(raw.get("status")),
            "age": _as_float(raw.get("age_at_diagnosis_years")),
            "raw_stage": str(raw.get("raw_stage") or "").strip(),
            "stage_group": str(raw.get("stage_group") or "").strip().lower(),
        }
    if not result:
        raise ValueError("reference cohort.csv is empty")
    return result


def _validate_png(payload: bytes) -> tuple[float, dict[str, object]]:
    signature = b"\x89PNG\r\n\x1a\n"
    if not payload.startswith(signature):
        raise ValueError("km_plot.png has an invalid PNG signature")
    offset = len(signature)
    width = height = None
    idat_parts: list[bytes] = []
    saw_iend = False
    while offset + 12 <= len(payload):
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        chunk_type = payload[offset + 4 : offset + 8]
        data_start = offset + 8
        data_end = data_start + length
        crc_end = data_end + 4
        if crc_end > len(payload):
            raise ValueError("km_plot.png contains a truncated PNG chunk")
        chunk_data = payload[data_start:data_end]
        expected_crc = struct.unpack(">I", payload[data_end:crc_end])[0]
        actual_crc = binascii.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise ValueError("km_plot.png contains an invalid PNG chunk checksum")
        if chunk_type == b"IHDR":
            if length != 13 or width is not None:
                raise ValueError("km_plot.png has an invalid IHDR chunk")
            width, height = struct.unpack(">II", chunk_data[:8])
        elif chunk_type == b"IDAT":
            idat_parts.append(chunk_data)
        elif chunk_type == b"IEND":
            saw_iend = True
            offset = crc_end
            break
        offset = crc_end
    if width is None or height is None or not saw_iend or not idat_parts:
        raise ValueError("km_plot.png is missing required PNG chunks")
    if width < 320 or height < 240 or width * height > 100_000_000:
        raise ValueError(f"km_plot.png has unreasonable dimensions: {width}x{height}")
    try:
        raw_pixels = zlib.decompress(b"".join(idat_parts))
    except zlib.error as exc:
        raise ValueError("km_plot.png has invalid compressed pixel data") from exc
    if len(raw_pixels) < height or len(set(raw_pixels)) < 4:
        raise ValueError("km_plot.png appears empty or single-color")

    quality = 1.0
    try:
        from PIL import Image, ImageStat

        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
        with Image.open(io.BytesIO(payload)) as image:
            rgb = image.convert("RGB")
            rgb.thumbnail((256, 256))
            extrema = rgb.getextrema()
            variation = max(high - low for low, high in extrema)
            standard_deviation = max(ImageStat.Stat(rgb).stddev)
            if variation < 8 or standard_deviation < 1.0:
                raise ValueError("km_plot.png has no meaningful visual variation")
    except ImportError:
        quality = 0.9
    details = {
        "km_plot_size_bytes": len(payload),
        "km_plot_width": width,
        "km_plot_height": height,
    }
    return quality, details


def _strip_r_comments(text: str) -> str:
    output: list[str] = []
    for line in text.splitlines():
        quote: str | None = None
        escaped = False
        kept: list[str] = []
        for char in line:
            if escaped:
                kept.append(char)
                escaped = False
                continue
            if char == "\\" and quote:
                kept.append(char)
                escaped = True
                continue
            if char in {'"', "'"}:
                if quote == char:
                    quote = None
                elif quote is None:
                    quote = char
                kept.append(char)
                continue
            if char == "#" and quote is None:
                break
            kept.append(char)
        output.append("".join(kept))
    return "\n".join(output)


def _fallback_r_syntax_check(code: str) -> bool:
    quote: str | None = None
    escaped = False
    stack: list[str] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for char in code:
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote:
            escaped = True
            continue
        if char in {'"', "'", "`"}:
            if quote == char:
                quote = None
            elif quote is None:
                quote = char
            continue
        if quote:
            continue
        if char in "([{":
            stack.append(char)
        elif char in ")]}":
            if not stack or stack.pop() != pairs[char]:
                return False
    return quote is None and not stack


def _validate_analysis_script(
    payload: bytes,
) -> tuple[float, bool, list[str], dict[str, object]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("analysis.R is not valid UTF-8") from exc
    code = _strip_r_comments(text)
    if not code.strip():
        return 0.0, False, ["analysis.R contains no executable code"], {}

    reference_pattern = re.compile(
        r"(?:read|load|unzip|untar|system|file|list\.files|dir)[A-Za-z0-9_.]*\s*\([^\n)]*"
        r"(?:reference\.7z|reference_outputs|evaluation_contract|/reference/)",
        re.IGNORECASE,
    )
    if reference_pattern.search(code):
        raise PermissionError(
            "analysis.R attempts to access evaluator-controlled reference data"
        )

    warnings: list[str] = []
    syntax_valid = _fallback_r_syntax_check(code)
    rscript = shutil.which("Rscript")
    if rscript:
        with tempfile.TemporaryDirectory(prefix="tcga-luad-r-parse-") as tmp:
            path = Path(tmp) / "analysis.R"
            path.write_text(text, encoding="utf-8")
            completed = subprocess.run(
                [
                    rscript,
                    "--vanilla",
                    "-e",
                    "parse(file=commandArgs(trailingOnly=TRUE)[1])",
                    str(path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            syntax_valid = completed.returncode == 0
            if not syntax_valid:
                warnings.append("R parser rejected analysis.R")
    else:
        warnings.append(
            "Rscript unavailable; analysis.R syntax used the balanced-delimiter fallback"
        )
    if not syntax_valid:
        raise ValueError("analysis.R is not syntactically valid")

    lowered = code.lower()
    checks = {
        "data_acquisition": bool(
            re.search(r"\bgdc(query|download|prepare)\s*\(", lowered)
            or (
                "api.gdc.cancer.gov" in lowered
                and re.search(r"\b(get|post|download\.file)\s*\(", lowered)
            )
        ),
        "expression_extraction": bool(
            ("kras" in lowered or "ensg00000133703" in lowered)
            and "tpm_unstranded" in lowered
            and re.search(
                r"\b(assay|rowdata|read[._]?(csv|tsv|delim|table)|fread)\s*\(", lowered
            )
        ),
        "cohort_construction": bool(
            re.search(
                r"\b(data\.frame|merge|left_join|inner_join|group_by|aggregate)\s*\(",
                lowered,
            )
        ),
        "survival_analysis": bool(
            re.search(r"\bsurv\s*\(", lowered)
            and re.search(r"\b(coxph|survfit|survdiff)\s*\(", lowered)
            and re.search(r"\bcox\.zph\s*\(", lowered)
        ),
        "outputs_written": bool(
            re.search(
                r"\b(write\.csv|write_csv|write_json|writejson|ggsave|png)\s*\(",
                lowered,
            )
        ),
    }
    workflow_score = sum(checks.values()) / len(checks)
    missing = [name for name, present in checks.items() if not present]
    if missing:
        warnings.append(f"analysis.R workflow evidence missing: {missing}")
    return workflow_score, syntax_valid, warnings, {"analysis_workflow_checks": checks}


def _median_and_groups(rows: list[dict[str, Any]]) -> tuple[float, list[str]]:
    median = float(statistics.median(row["expression"] for row in rows))
    groups = ["high" if row["expression"] >= median else "low" for row in rows]
    return median, groups


def _summary_consistency(
    data: dict[str, Any], rows: list[dict[str, Any]], median: float
) -> tuple[float, bool, dict[str, object]]:
    summary = data.get("cohort_summary")
    if not isinstance(summary, dict):
        return 0.0, False, {}
    expected = {
        "n_patients": len(rows),
        "n_events": sum(row["status"] for row in rows),
        "n_kras_high": sum(row["group"] == "high" for row in rows),
        "n_kras_low": sum(row["group"] == "low" for row in rows),
    }
    scores = [
        1.0 if _as_int(summary.get(name)) == value else 0.0
        for name, value in expected.items()
    ]
    median_score = _close_score(
        _as_float(summary.get("median_kras_expression")),
        median,
        atol=1e-6,
        rtol=1e-4,
    )
    scores.append(median_score)
    return sum(scores) / len(scores), all(score == 1.0 for score in scores), expected


def _logrank(rows: list[dict[str, Any]]) -> tuple[float, float]:
    event_times = sorted({row["time_days"] for row in rows if row["status"] == 1})
    observed_high = 0.0
    expected_high = 0.0
    variance = 0.0
    for time in event_times:
        at_risk = [row for row in rows if row["time_days"] >= time]
        events = [
            row for row in rows if row["time_days"] == time and row["status"] == 1
        ]
        n = len(at_risk)
        d = len(events)
        if n <= 1 or d == 0:
            continue
        n_high = sum(row["group"] == "high" for row in at_risk)
        d_high = sum(row["group"] == "high" for row in events)
        observed_high += d_high
        expected_high += d * n_high / n
        variance += d * (n_high / n) * (1.0 - n_high / n) * ((n - d) / (n - 1.0))
    statistic = (observed_high - expected_high) ** 2 / variance if variance > 0 else 0.0
    p_value = math.erfc(math.sqrt(max(statistic, 0.0) / 2.0))
    return statistic, p_value


def _solve_linear(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [matrix[row][:] + [vector[row]] for row in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ArithmeticError("singular Cox information matrix")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                augmented[row][index] - factor * augmented[column][index]
                for index in range(size + 1)
            ]
    return [augmented[row][-1] for row in range(size)]


def _invert_matrix(matrix: list[list[float]]) -> list[list[float]]:
    size = len(matrix)
    columns = []
    for column in range(size):
        unit = [1.0 if row == column else 0.0 for row in range(size)]
        columns.append(_solve_linear(matrix, unit))
    return [[columns[column][row] for column in range(size)] for row in range(size)]


def _cox_efron(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    raw_x = [
        [
            1.0 if row["group"] == "high" else 0.0,
            row["age"],
            1.0 if row["stage_group"] == "advanced" else 0.0,
        ]
        for row in rows
    ]
    means = [sum(row[column] for row in raw_x) / len(raw_x) for column in range(3)]
    x = [[row[column] - means[column] for column in range(3)] for row in raw_x]
    event_times = sorted({row["time_days"] for row in rows if row["status"] == 1})
    beta = [0.0, 0.0, 0.0]
    final_info: list[list[float]] | None = None

    for _ in range(60):
        eta = [sum(beta[j] * vector[j] for j in range(3)) for vector in x]
        exp_eta = [math.exp(max(-50.0, min(50.0, value))) for value in eta]
        gradient = [0.0, 0.0, 0.0]
        info = [[0.0 for _ in range(3)] for _ in range(3)]
        for time in event_times:
            event_indices = [
                index
                for index, row in enumerate(rows)
                if row["time_days"] == time and row["status"] == 1
            ]
            if not event_indices:
                continue
            risk_indices = [
                index for index, row in enumerate(rows) if row["time_days"] >= time
            ]
            deaths = len(event_indices)

            s0 = sum(exp_eta[index] for index in risk_indices)
            s1 = [
                sum(exp_eta[index] * x[index][j] for index in risk_indices)
                for j in range(3)
            ]
            s2 = [
                [
                    sum(
                        exp_eta[index] * x[index][j] * x[index][k]
                        for index in risk_indices
                    )
                    for k in range(3)
                ]
                for j in range(3)
            ]
            e0 = sum(exp_eta[index] for index in event_indices)
            e1 = [
                sum(exp_eta[index] * x[index][j] for index in event_indices)
                for j in range(3)
            ]
            e2 = [
                [
                    sum(
                        exp_eta[index] * x[index][j] * x[index][k]
                        for index in event_indices
                    )
                    for k in range(3)
                ]
                for j in range(3)
            ]
            for index in event_indices:
                for j in range(3):
                    gradient[j] += x[index][j]
            for tie_index in range(deaths):
                fraction = tie_index / deaths
                denominator = s0 - fraction * e0
                if denominator <= 0:
                    raise ArithmeticError("invalid Cox risk-set denominator")
                mean = [(s1[j] - fraction * e1[j]) / denominator for j in range(3)]
                for j in range(3):
                    gradient[j] -= mean[j]
                    for k in range(3):
                        second = (s2[j][k] - fraction * e2[j][k]) / denominator
                        info[j][k] += second - mean[j] * mean[k]
        final_info = info
        delta = _solve_linear(info, gradient)
        beta = [beta[index] + delta[index] for index in range(3)]
        if max(abs(value) for value in delta) < 1e-9:
            break
    if final_info is None:
        raise ArithmeticError("Cox model has no events")

    # Recompute the information matrix at the converged coefficients for SEs.
    eta = [sum(beta[j] * vector[j] for j in range(3)) for vector in x]
    exp_eta = [math.exp(max(-50.0, min(50.0, value))) for value in eta]
    info = [[0.0 for _ in range(3)] for _ in range(3)]
    for time in event_times:
        event_indices = [
            index
            for index, row in enumerate(rows)
            if row["time_days"] == time and row["status"] == 1
        ]
        risk_indices = [
            index for index, row in enumerate(rows) if row["time_days"] >= time
        ]
        deaths = len(event_indices)
        s0 = sum(exp_eta[index] for index in risk_indices)
        s1 = [
            sum(exp_eta[index] * x[index][j] for index in risk_indices)
            for j in range(3)
        ]
        s2 = [
            [
                sum(
                    exp_eta[index] * x[index][j] * x[index][k] for index in risk_indices
                )
                for k in range(3)
            ]
            for j in range(3)
        ]
        e0 = sum(exp_eta[index] for index in event_indices)
        e1 = [
            sum(exp_eta[index] * x[index][j] for index in event_indices)
            for j in range(3)
        ]
        e2 = [
            [
                sum(
                    exp_eta[index] * x[index][j] * x[index][k]
                    for index in event_indices
                )
                for k in range(3)
            ]
            for j in range(3)
        ]
        for tie_index in range(deaths):
            fraction = tie_index / deaths
            denominator = s0 - fraction * e0
            mean = [(s1[j] - fraction * e1[j]) / denominator for j in range(3)]
            for j in range(3):
                for k in range(3):
                    second = (s2[j][k] - fraction * e2[j][k]) / denominator
                    info[j][k] += second - mean[j] * mean[k]
    covariance = _invert_matrix(info)
    output: dict[str, dict[str, float]] = {}
    for index, variable in enumerate(EXPECTED_COX_VARIABLES):
        standard_error = math.sqrt(max(covariance[index][index], 0.0))
        z_value = beta[index] / standard_error if standard_error else 0.0
        p_value = math.erfc(abs(z_value) / math.sqrt(2.0))
        output[variable] = {
            "hazard_ratio": math.exp(beta[index]),
            "p_value": p_value,
            "ci_lower_95": math.exp(beta[index] - 1.959963984540054 * standard_error),
            "ci_upper_95": math.exp(beta[index] + 1.959963984540054 * standard_error),
        }
    return output


def _normalize_variable_name(value: object) -> str | None:
    name = re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")
    if name == "global":
        return "GLOBAL"
    if "kras" in name:
        return "kras_group_high"
    if "stage" in name:
        return "stage_group_advanced"
    if "age" in name:
        return "age_at_diagnosis_years"
    return None


def _normalize_coefficients(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cox_model = data.get("cox_model")
    if not isinstance(cox_model, dict):
        return {}
    raw = cox_model.get("coefficients")
    items: list[tuple[object, object]] = []
    if isinstance(raw, dict):
        items = list(raw.items())
    elif isinstance(raw, list):
        items = [(item.get("variable"), item) for item in raw if isinstance(item, dict)]
    result: dict[str, dict[str, Any]] = {}
    for raw_name, raw_values in items:
        if not isinstance(raw_values, dict):
            continue
        name = _normalize_variable_name(raw_name or raw_values.get("variable"))
        if name is None:
            continue
        result[name] = {
            "hazard_ratio": _first_float(
                raw_values, "hazard_ratio", "hr", "exp_coef", "exp(coef)"
            ),
            "p_value": _first_float(raw_values, "p_value", "p", "pvalue"),
            "ci_lower_95": _first_float(
                raw_values, "ci_lower_95", "hr_lower95", "lower_95", "lower_95%"
            ),
            "ci_upper_95": _first_float(
                raw_values, "ci_upper_95", "hr_upper95", "upper_95", "upper_95%"
            ),
        }
    return result


def _first_float(mapping: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in mapping:
            value = _as_float(mapping.get(key))
            if value is not None:
                return value
    return None


def _logrank_values(data: dict[str, Any]) -> tuple[float | None, float | None]:
    section = data.get("log_rank")
    if not isinstance(section, dict):
        return None, None
    statistic = _first_float(
        section, "test_statistic", "statistic", "chisq", "chi_square"
    )
    p_value = _first_float(section, "p_value", "p", "pvalue")
    return statistic, p_value


def _validate_ph_test(data: dict[str, Any]) -> tuple[bool, dict[str, object]]:
    section = data.get("ph_test")
    if not isinstance(section, dict):
        return False, {"ph_variables": []}
    entries: list[tuple[object, object]] = []
    if isinstance(section.get("per_variable"), dict):
        entries.extend(section["per_variable"].items())
    if isinstance(section.get("table"), list):
        entries.extend(
            (item.get("variable"), item)
            for item in section["table"]
            if isinstance(item, dict)
        )
    if not entries:
        entries.extend(
            (key, value)
            for key, value in section.items()
            if isinstance(value, dict) and _normalize_variable_name(key)
        )
    valid_variables: set[str] = set()
    for raw_name, raw_values in entries:
        if not isinstance(raw_values, dict):
            continue
        name = _normalize_variable_name(raw_name or raw_values.get("variable"))
        if name is None:
            continue
        statistic = _first_float(
            raw_values, "test_statistic", "statistic", "chisq", "chi_square"
        )
        p_value = _first_float(raw_values, "p_value", "p", "pvalue")
        if (
            statistic is None
            or statistic < 0
            or p_value is None
            or not 0 <= p_value <= 1
        ):
            continue
        valid_variables.add(name)
    expected = set(EXPECTED_COX_VARIABLES)
    valid = expected.issubset(valid_variables)
    return valid, {"ph_variables": sorted(valid_variables)}


def _reference_alignment(
    rows: list[dict[str, Any]], reference: dict[str, dict[str, Any]]
) -> tuple[float, float, float, dict[str, object], list[str]]:
    observed = {row["patient_id"]: row for row in rows}
    common_ids = sorted(set(observed) & set(reference))
    patient_f1 = 2.0 * len(common_ids) / (len(observed) + len(reference))
    if common_ids:
        sample_match = sum(
            observed[patient_id]["sample_id"] == reference[patient_id]["sample_id"]
            for patient_id in common_ids
        ) / len(common_ids)
        file_match = sum(
            observed[patient_id]["file_id"] == reference[patient_id]["file_id"]
            for patient_id in common_ids
        ) / len(common_ids)
    else:
        sample_match = file_match = 0.0

    clinical_checks: list[float] = []
    verifiable: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for patient_id in common_ids:
        row = observed[patient_id]
        ref = reference[patient_id]
        clinical_checks.extend(
            [
                1.0 if row["status"] == ref["status"] else 0.0,
                _close_score(row["time_days"], ref["time_days"], atol=1.0),
                _close_score(row["age"], ref["age"], atol=0.55),
                1.0 if row["stage_group"] == ref["stage_group"] else 0.0,
            ]
        )
        if row["file_id"] == ref["file_id"] and ref["expression"] is not None:
            verifiable.append((row, ref))
    clinical_match = (
        sum(clinical_checks) / len(clinical_checks) if clinical_checks else 0.0
    )
    expression_coverage = len(verifiable) / len(rows)
    expression_scores = [
        _close_score(row["expression"], ref["expression"], atol=1e-4, rtol=1e-4)
        for row, ref in verifiable
    ]
    expression_match = (
        sum(score == 1.0 for score in expression_scores) / len(expression_scores)
        if expression_scores
        else 0.0
    )

    cohort_score = (
        0.40 * _band_score(patient_f1, zero_below=0.80, full_at=0.95)
        + 0.15 * _band_score(sample_match, zero_below=0.80, full_at=0.95)
        + 0.15 * _band_score(file_match, zero_below=0.80, full_at=0.95)
        + 0.30 * _band_score(clinical_match, zero_below=0.85, full_at=0.97)
    )
    expression_score = 0.5 * _band_score(
        expression_coverage, zero_below=0.80, full_at=0.90
    ) + 0.5 * _band_score(expression_match, zero_below=0.95, full_at=0.98)
    metrics: dict[str, object] = {
        "submitted_patients": len(rows),
        "reference_patients": len(reference),
        "common_patients": len(common_ids),
        "patient_f1": patient_f1,
        "sample_match_rate": sample_match,
        "file_match_rate": file_match,
        "clinical_match_rate": clinical_match,
        "expression_verification_coverage": expression_coverage,
        "expression_match_rate": expression_match,
    }
    messages: list[str] = []
    if patient_f1 < 0.95:
        messages.append(f"patient-set F1 below full-credit threshold: {patient_f1:.3f}")
    if file_match < 0.95:
        messages.append(
            f"reference file-ID match below full-credit threshold: {file_match:.3f}"
        )
    if expression_coverage < 0.90:
        messages.append(
            f"KRAS expression verification coverage is {expression_coverage:.3f}"
        )
    if expression_match < 0.98:
        messages.append(
            f"verified KRAS expression match rate is {expression_match:.3f}"
        )
    return cohort_score, expression_score, expression_match, metrics, messages


def _statistics_score(
    data: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    summary_score: float,
) -> tuple[float, bool, dict[str, object], list[str]]:
    messages: list[str] = []
    expected_statistic, expected_p = _logrank(rows)
    observed_statistic, observed_p = _logrank_values(data)
    logrank_score = 0.5 * _close_score(
        observed_statistic, expected_statistic, atol=1e-3, rtol=0.01
    ) + 0.5 * _close_score(observed_p, expected_p, atol=0.01)
    if logrank_score < 1.0:
        messages.append("reported log-rank result differs from recomputation")

    expected_cox = _cox_efron(rows)
    observed_cox = _normalize_coefficients(data)
    coefficient_scores: list[float] = []
    for variable in EXPECTED_COX_VARIABLES:
        expected = expected_cox[variable]
        observed = observed_cox.get(variable, {})
        for field in ("hazard_ratio", "ci_lower_95", "ci_upper_95"):
            coefficient_scores.append(
                _close_score(observed.get(field), expected[field], atol=1e-4, rtol=0.02)
            )
        coefficient_scores.append(
            _close_score(observed.get("p_value"), expected["p_value"], atol=0.01)
        )
    cox_score = sum(coefficient_scores) / len(coefficient_scores)
    if cox_score < 1.0:
        messages.append(
            "reported Cox coefficients differ from Efron-ties recomputation"
        )

    ph_valid, ph_details = _validate_ph_test(data)
    if not ph_valid:
        messages.append(
            "ph_test is missing required variables or contains invalid values"
        )
    total = (
        0.15 * summary_score
        + 0.25 * logrank_score
        + 0.50 * cox_score
        + 0.10 * float(ph_valid)
    )
    details: dict[str, object] = {
        "recomputed_log_rank": {
            "test_statistic": expected_statistic,
            "p_value": expected_p,
        },
        "recomputed_cox": expected_cox,
        "log_rank_consistency": logrank_score,
        "cox_consistency": cox_score,
        "ph_test_valid": ph_valid,
        **ph_details,
    }
    return total, ph_valid, details, messages


def score_submission(
    outputs: dict[str, bytes],
    *,
    reference_cohort_csv: bytes,
    reference_cox_json: bytes,
    evaluation_contract_json: bytes,
) -> ScoreResult:
    """Score one submission without requiring exact reproduction of mutable GDC data."""

    del reference_cox_json  # v2 recomputes statistics from the submitted cohort.
    missing = [name for name in REQUIRED_FILES if not outputs.get(name)]
    if missing:
        return _hard_failure(f"missing or empty required files: {missing}")

    try:
        contract = _load_json(evaluation_contract_json, "evaluation_contract.json")
        minimum_patient_count = int(contract.get("minimum_patient_count", 400))
        rows, fieldnames = _parse_cohort(
            outputs["cohort.csv"], minimum_patient_count=minimum_patient_count
        )
        reference = _reference_rows(reference_cohort_csv)
        data = _load_json(outputs["cox_results.json"], "cox_results.json")
        png_quality, png_details = _validate_png(outputs["km_plot.png"])
        (
            workflow_score,
            syntax_valid,
            script_warnings,
            script_details,
        ) = _validate_analysis_script(outputs["analysis.R"])
    except PermissionError as exc:
        return _hard_failure(str(exc))
    except Exception as exc:
        return _hard_failure(str(exc))

    reasons: list[str] = []
    warnings = list(script_warnings)
    details: dict[str, object] = {
        "cohort_rows": len(rows),
        "cohort_columns": fieldnames,
        **png_details,
        **script_details,
    }

    required_json_sections = ("cohort_summary", "log_rank", "cox_model", "ph_test")
    present_json_sections = sum(
        isinstance(data.get(name), dict) for name in required_json_sections
    )
    json_schema_score = present_json_sections / len(required_json_sections)
    if json_schema_score < 1.0:
        reasons.append("cox_results.json is missing one or more required sections")
    artifacts_score = (1.0 + json_schema_score + 1.0 + float(syntax_valid)) / 4.0

    (
        cohort_score,
        expression_score,
        expression_match,
        alignment_metrics,
        alignment_messages,
    ) = _reference_alignment(rows, reference)
    warnings.extend(alignment_messages)
    details.update(alignment_metrics)

    median, expected_groups = _median_and_groups(rows)
    group_match_rate = sum(
        row["group"] == expected for row, expected in zip(rows, expected_groups)
    ) / len(rows)
    stage_checks = [
        row["stage_group"] == normalized
        for row in rows
        if (normalized := _normalize_stage(row["raw_stage"])) is not None
    ]
    stage_match_rate = sum(stage_checks) / len(stage_checks) if stage_checks else 0.5
    summary_score, summary_consistent, expected_summary = _summary_consistency(
        data, rows, median
    )
    derivations_score = (
        0.40 * expression_score
        + 0.25 * group_match_rate
        + 0.20 * summary_score
        + 0.15 * stage_match_rate
    )
    details.update(
        {
            "recomputed_median_kras_expression": median,
            "group_label_match_rate": group_match_rate,
            "stage_mapping_match_rate": stage_match_rate,
            "summary_consistent": summary_consistent,
            "recomputed_summary": expected_summary,
        }
    )
    if group_match_rate < 1.0:
        reasons.append(
            f"{sum(row['group'] != expected for row, expected in zip(rows, expected_groups))} KRAS group labels disagree with the submitted-cohort median"
        )
    if not summary_consistent:
        reasons.append(
            "cox_results.json cohort_summary is inconsistent with cohort.csv"
        )
    if stage_match_rate < 1.0:
        warnings.append("one or more recognizable raw stages disagree with stage_group")

    try:
        (
            statistics_score,
            ph_valid,
            statistics_details,
            statistics_messages,
        ) = _statistics_score(data, rows, summary_score=summary_score)
    except Exception as exc:
        return _hard_failure(
            f"evaluator could not recompute survival statistics: {exc}", details=details
        )
    details.update(statistics_details)
    reasons.extend(statistics_messages)

    reproducibility_score = 0.75 * workflow_score + 0.25 * png_quality
    if workflow_score < 0.8:
        reasons.append(
            "analysis.R does not provide enough executable workflow evidence"
        )

    sections = {
        "artifacts": max(0.0, min(1.0, artifacts_score)),
        "cohort": max(0.0, min(1.0, cohort_score)),
        "derivations": max(0.0, min(1.0, derivations_score)),
        "statistics": max(0.0, min(1.0, statistics_score)),
        "reproducibility": max(0.0, min(1.0, reproducibility_score)),
    }
    score = sum(SECTION_WEIGHTS[name] * sections[name] for name in SECTION_WEIGHTS)

    expression_coverage = float(alignment_metrics["expression_verification_coverage"])
    patient_f1 = float(alignment_metrics["patient_f1"])
    critical_checks = {
        "patient_set_similarity": patient_f1 >= 0.95,
        "expression_verification_coverage": expression_coverage >= 0.80,
        "expression_match_rate": expression_match >= 0.95,
        "summary_consistent": summary_consistent,
        "ph_test_valid": ph_valid,
        "group_labels_consistent": group_match_rate >= 0.995,
    }
    details["critical_checks"] = critical_checks
    for name, passed_check in critical_checks.items():
        if not passed_check:
            reasons.append(f"critical check failed: {name}")

    quality_gates_passed = (
        sections["artifacts"] >= 0.80
        and sections["derivations"] >= 0.80
        and sections["statistics"] >= 0.60
        and sections["reproducibility"] >= 0.50
        and all(critical_checks.values())
    )
    uncapped_score = score
    if not quality_gates_passed:
        # The benchmark runner persists only the numeric score.  Keep useful
        # partial credit, but make critical semantic failures visibly fail the
        # advertised 0.85 threshold even when other sections are excellent.
        score = min(score, 0.84)
    passed = score >= 0.85 and quality_gates_passed
    details["pass_threshold"] = 0.85
    details["uncapped_score"] = uncapped_score
    details["section_weights"] = SECTION_WEIGHTS
    return ScoreResult(
        score=round(score, 6),
        passed=passed,
        reasons=list(dict.fromkeys(reasons)),
        details=details,
        sections={name: round(value, 6) for name, value in sections.items()},
        warnings=list(dict.fromkeys(warnings)),
        hard_failure=False,
    )
