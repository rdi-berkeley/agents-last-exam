"""Scorer for TCGA BRCA differential expression outputs."""

from __future__ import annotations

import csv
import io
import json
import math
import re
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any


REQUIRED_FILES = [
    "sample_summary.txt",
    "pca_plot.png",
    "deg_results_all.csv",
    "deg_results_significant.csv",
    "volcano_plot.png",
    "benchmark_summary.csv",
    "run_manifest.json",
]

RESULT_COLUMNS = ["gene", "mean_tumor", "mean_normal", "log2FC", "t_statistic", "pvalue", "padj"]
MIN_IMAGE_BYTES = 10_000
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PASS_THRESHOLDS = {
    "recall_up": 0.80,
    "recall_down": 0.80,
    "precision_up": 0.70,
    "precision_down": 0.70,
    "f1_overall": 0.75,
}


@dataclass
class ScoreReport:
    score: float = 0.0
    passed: bool = False
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def fail(self, message: str) -> None:
        self.failures.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def finish(self) -> "ScoreReport":
        self.passed = not self.failures
        self.score = 1.0 if self.passed else 0.0
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "passed": self.passed,
            "failures": self.failures,
            "warnings": self.warnings,
            "metrics": self.metrics,
        }


def _decode(payload: bytes, filename: str) -> str:
    try:
        return payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{filename} is not valid UTF-8 text") from exc


def _read_csv(payload: bytes, filename: str) -> tuple[list[dict[str, str]], list[str]]:
    text = _decode(payload, filename)
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    return rows, list(reader.fieldnames or [])


def _norm_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _column_map(fieldnames: list[str]) -> dict[str, str]:
    normalized = {_norm_col(name): name for name in fieldnames}
    mapping: dict[str, str] = {}
    aliases = {
        "gene": ["gene", "genesymbol", "symbol"],
        "mean_tumor": ["meantumor", "tumormean", "meanprimarytumor"],
        "mean_normal": ["meannormal", "normalmean", "meansolidtissuenormal"],
        "log2FC": ["log2fc", "logfc", "log2foldchange"],
        "t_statistic": ["tstatistic", "tstat", "statistic"],
        "pvalue": ["pvalue", "pval", "p"],
        "padj": ["padj", "adjustedpvalue", "fdr", "bh"],
    }
    for canonical, options in aliases.items():
        for option in options:
            if option in normalized:
                mapping[canonical] = normalized[option]
                break
    return mapping


def _float(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = str(value).strip()
    if stripped.lower() in {"", "na", "nan", "none", "null", "inf", "-inf"}:
        return None
    try:
        parsed = float(stripped)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _validate_png(payload: bytes, filename: str) -> tuple[bool, str | None, dict[str, int]]:
    if len(payload) < MIN_IMAGE_BYTES:
        return False, f"{filename} is too small: {len(payload)} bytes", {}
    if not payload.startswith(PNG_SIGNATURE):
        return False, f"{filename} is not a valid PNG", {}

    offset = len(PNG_SIGNATURE)
    saw_ihdr = False
    saw_idat = False
    saw_iend = False
    total_idat_bytes = 0
    dimensions: dict[str, int] = {}
    while offset + 12 <= len(payload):
        chunk_length = struct.unpack(">I", payload[offset : offset + 4])[0]
        chunk_type = payload[offset + 4 : offset + 8]
        chunk_data_start = offset + 8
        chunk_data_end = chunk_data_start + chunk_length
        crc_start = chunk_data_end
        crc_end = crc_start + 4
        if crc_end > len(payload):
            return False, f"{filename} has a truncated PNG chunk", dimensions
        expected_crc = struct.unpack(">I", payload[crc_start:crc_end])[0]
        actual_crc = zlib.crc32(chunk_type + payload[chunk_data_start:chunk_data_end]) & 0xFFFFFFFF
        if expected_crc != actual_crc:
            return False, f"{filename} has an invalid PNG chunk CRC", dimensions
        if chunk_type == b"IHDR":
            if saw_ihdr or chunk_length != 13:
                return False, f"{filename} has an invalid PNG IHDR chunk", dimensions
            saw_ihdr = True
            width, height = struct.unpack(">II", payload[chunk_data_start : chunk_data_start + 8])
            dimensions = {"width": width, "height": height}
            if width < 400 or height < 300:
                return False, f"{filename} dimensions too small: {width}x{height}", dimensions
        elif chunk_type == b"IDAT":
            saw_idat = True
            total_idat_bytes += chunk_length
        elif chunk_type == b"IEND":
            saw_iend = True
            if crc_end != len(payload):
                return False, f"{filename} has trailing bytes after IEND", dimensions
            break
        offset = crc_end

    if not saw_ihdr:
        return False, f"{filename} is missing IHDR", dimensions
    if not saw_idat or total_idat_bytes < 1_000:
        return False, f"{filename} has no substantial IDAT image data", dimensions
    if not saw_iend:
        return False, f"{filename} is missing IEND", dimensions
    return True, None, dimensions


def _parse_sample_counts(text: str) -> tuple[int | None, int | None]:
    tumor_count: int | None = None
    normal_count: int | None = None
    for line in text.lower().splitlines():
        numbers = [int(value) for value in re.findall(r"\d+", line)]
        if not numbers:
            continue
        if "tumor" in line and "normal" not in line:
            tumor_count = next((value for value in numbers if 500 <= value <= 1500), tumor_count)
        elif "normal" in line and "tumor" not in line:
            normal_count = next((value for value in numbers if 50 <= value <= 300), normal_count)
        elif "tumor" in line and "normal" in line:
            for value in numbers:
                if 500 <= value <= 1500 and tumor_count is None:
                    tumor_count = value
                elif 50 <= value <= 300 and normal_count is None:
                    normal_count = value
    return tumor_count, normal_count


def _truth_by_direction(reference_truth_csv: bytes) -> dict[str, dict[str, str]]:
    rows, fields = _read_csv(reference_truth_csv, "reference/truth_breast_degs.csv")
    mapping = _column_map(fields)
    gene_col = mapping.get("gene", "gene")
    direction_col = "expected_direction"
    truth: dict[str, dict[str, str]] = {"up": {}, "down": {}}
    for row in rows:
        gene = (row.get(gene_col) or row.get("gene") or "").strip().upper()
        direction = (row.get(direction_col) or "").strip().lower()
        if gene and direction in truth:
            truth[direction][gene] = direction
    return truth


def _truth_metrics(
    significant_rows: list[dict[str, str]],
    sig_cols: dict[str, str],
    truth: dict[str, dict[str, str]],
) -> dict[str, float]:
    gene_col = sig_cols["gene"]
    fc_col = sig_cols["log2FC"]
    significant_by_gene: dict[str, float] = {}
    for row in significant_rows:
        gene = (row.get(gene_col) or "").strip().upper()
        log2fc = _float(row.get(fc_col))
        if gene and log2fc is not None:
            significant_by_gene[gene] = log2fc

    values: dict[str, float] = {}
    all_correct = 0
    all_detected = 0
    all_truth = 0
    for direction, genes in truth.items():
        correct = 0
        detected = 0
        expected_sign = 1 if direction == "up" else -1
        for gene in genes:
            if gene not in significant_by_gene:
                continue
            detected += 1
            if significant_by_gene[gene] * expected_sign > 0:
                correct += 1
        total = len(genes)
        recall = correct / total if total else 0.0
        precision = correct / detected if detected else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        values[f"recall_{direction}"] = recall
        values[f"precision_{direction}"] = precision
        values[f"f1_{direction}"] = f1
        all_correct += correct
        all_detected += detected
        all_truth += total

    precision_overall = all_correct / all_detected if all_detected else 0.0
    recall_overall = all_correct / all_truth if all_truth else 0.0
    f1 = (
        2 * precision_overall * recall_overall / (precision_overall + recall_overall)
        if precision_overall + recall_overall
        else 0.0
    )
    values["precision_overall"] = precision_overall
    values["recall_overall"] = recall_overall
    values["f1_overall"] = f1
    return values


def _validate_benchmark_summary(payload: bytes) -> tuple[list[str], dict[str, dict[str, float]]]:
    failures: list[str] = []
    rows, fields = _read_csv(payload, "benchmark_summary.csv")
    normalized_fields = {_norm_col(field): field for field in fields}
    required = ["metric", "up", "down", "overall"]
    missing = [name for name in required if name not in normalized_fields]
    if missing:
        return ["benchmark_summary.csv missing columns: " + ", ".join(missing)], {}

    metric_col = normalized_fields["metric"]
    parsed: dict[str, dict[str, float]] = {}
    for row in rows:
        metric = (row.get(metric_col) or "").strip().lower()
        if metric:
            parsed[metric] = {}
            for direction in ["up", "down", "overall"]:
                value = _float(row.get(normalized_fields[direction]))
                if value is None:
                    failures.append(f"benchmark_summary.csv {metric}/{direction} is not numeric")
                else:
                    parsed[metric][direction] = value
    for metric in ["precision", "recall", "f1"]:
        if metric not in parsed:
            failures.append(f"benchmark_summary.csv missing metric row: {metric}")
    return failures, parsed


def _validate_significant_rows(
    significant_rows: list[dict[str, str]],
    sig_cols: dict[str, str],
    deg_rows: list[dict[str, str]],
    deg_cols: dict[str, str],
) -> list[str]:
    failures: list[str] = []
    if not {"gene", "log2FC", "padj", "pvalue"}.issubset(sig_cols):
        return failures
    invalid_threshold_examples: list[str] = []
    for row in significant_rows:
        gene = (row.get(sig_cols["gene"]) or "").strip()
        log2fc = _float(row.get(sig_cols["log2FC"]))
        padj = _float(row.get(sig_cols["padj"]))
        pvalue = _float(row.get(sig_cols["pvalue"]))
        if log2fc is None or padj is None or pvalue is None:
            invalid_threshold_examples.append(gene or "<blank>")
            continue
        if abs(log2fc) <= 1.0 or padj >= 0.05:
            invalid_threshold_examples.append(gene or "<blank>")
        if not (0.0 <= pvalue <= 1.0) or not (0.0 <= padj <= 1.0):
            invalid_threshold_examples.append(gene or "<blank>")
    if invalid_threshold_examples:
        preview = ", ".join(invalid_threshold_examples[:10])
        failures.append(
            "deg_results_significant.csv contains rows that do not satisfy "
            f"abs(log2FC) > 1 and padj < 0.05: {preview}"
        )

    if not {"gene", "log2FC", "padj"}.issubset(deg_cols):
        return failures
    deg_by_gene = {
        (row.get(deg_cols["gene"]) or "").strip().upper(): row
        for row in deg_rows
        if (row.get(deg_cols["gene"]) or "").strip()
    }
    missing_from_all = 0
    mismatched = 0
    checked = 0
    for row in significant_rows:
        gene = (row.get(sig_cols["gene"]) or "").strip().upper()
        if not gene:
            continue
        all_row = deg_by_gene.get(gene)
        if all_row is None:
            missing_from_all += 1
            continue
        checked += 1
        sig_fc = _float(row.get(sig_cols["log2FC"]))
        all_fc = _float(all_row.get(deg_cols["log2FC"]))
        sig_padj = _float(row.get(sig_cols["padj"]))
        all_padj = _float(all_row.get(deg_cols["padj"]))
        if sig_fc is None or all_fc is None or sig_padj is None or all_padj is None:
            mismatched += 1
            continue
        padj_tolerance = max(1e-12, abs(all_padj) * 1e-4)
        if abs(sig_fc - all_fc) > 1e-6 or abs(sig_padj - all_padj) > padj_tolerance:
            mismatched += 1
    if missing_from_all:
        failures.append(
            f"deg_results_significant.csv contains {missing_from_all} genes absent from deg_results_all.csv"
        )
    if checked and mismatched / checked > 0.01:
        failures.append(
            f"deg_results_significant.csv values mismatch deg_results_all.csv for {mismatched}/{checked} genes"
        )
    return failures


def _gold_sanity(
    deg_rows: list[dict[str, str]],
    deg_cols: dict[str, str],
    gold_rows: list[dict[str, str]],
    gold_cols: dict[str, str],
    truth: dict[str, dict[str, str]],
) -> tuple[int, int, float]:
    agent_by_gene = {
        (row.get(deg_cols["gene"]) or "").strip().upper(): row
        for row in deg_rows
        if (row.get(deg_cols["gene"]) or "").strip()
    }
    gold_by_gene = {
        (row.get(gold_cols["gene"]) or "").strip().upper(): row
        for row in gold_rows
        if (row.get(gold_cols["gene"]) or "").strip()
    }
    anchor_genes = list(truth["up"]) + list(truth["down"])
    anchor_genes.extend([row.get(gold_cols["gene"], "").strip().upper() for row in gold_rows[:75]])
    seen: set[str] = set()
    checked = 0
    matched = 0
    max_abs_log2fc_delta = 0.0
    for gene in anchor_genes:
        if not gene or gene in seen or gene not in gold_by_gene:
            continue
        seen.add(gene)
        checked += 1
        agent_row = agent_by_gene.get(gene)
        if agent_row is None:
            continue
        agent_fc = _float(agent_row.get(deg_cols["log2FC"]))
        gold_fc = _float(gold_by_gene[gene].get(gold_cols["log2FC"]))
        if agent_fc is None or gold_fc is None:
            continue
        delta = abs(agent_fc - gold_fc)
        max_abs_log2fc_delta = max(max_abs_log2fc_delta, delta)
        if delta <= 0.10:
            matched += 1
    return checked, matched, max_abs_log2fc_delta


def score_submission(
    outputs: dict[str, bytes],
    *,
    reference_truth_csv: bytes,
    gold_deg_results_all_csv: bytes | None = None,
) -> ScoreReport:
    report = ScoreReport()

    for filename in REQUIRED_FILES:
        if filename not in outputs:
            report.fail(f"missing required file: {filename}")
    if report.failures:
        return report.finish()

    for image_name in ["pca_plot.png", "volcano_plot.png"]:
        valid, failure, dimensions = _validate_png(outputs[image_name], image_name)
        report.metrics[f"{image_name}_dimensions"] = dimensions
        if not valid and failure:
            report.fail(failure)

    try:
        deg_rows, deg_fields = _read_csv(outputs["deg_results_all.csv"], "deg_results_all.csv")
        deg_cols = _column_map(deg_fields)
    except Exception as exc:
        report.fail(f"deg_results_all.csv parse failed: {exc}")
        deg_rows, deg_cols = [], {}

    missing_cols = [col for col in RESULT_COLUMNS if col not in deg_cols]
    if missing_cols:
        report.fail("deg_results_all.csv missing columns: " + ", ".join(missing_cols))
    if len(deg_rows) < 15_000:
        report.fail(f"deg_results_all.csv has {len(deg_rows)} rows, expected >= 15000")
    if len(deg_rows) < 18_000:
        report.fail(f"genes tested = {len(deg_rows)}, expected >= 18000")

    if "padj" in deg_cols and deg_rows:
        non_na_padj = sum(1 for row in deg_rows if _float(row.get(deg_cols["padj"])) is not None)
        padj_rate = non_na_padj / len(deg_rows)
        report.metrics["padj_non_na_rate"] = padj_rate
        if padj_rate < 0.90:
            report.fail(f"padj non-NA rate {padj_rate:.1%}, expected >= 90%")

    try:
        sig_rows, sig_fields = _read_csv(
            outputs["deg_results_significant.csv"], "deg_results_significant.csv"
        )
        sig_cols = _column_map(sig_fields)
    except Exception as exc:
        report.fail(f"deg_results_significant.csv parse failed: {exc}")
        sig_rows, sig_cols = [], {}

    sig_missing_cols = [col for col in RESULT_COLUMNS if col not in sig_cols]
    if sig_missing_cols:
        report.fail("deg_results_significant.csv missing columns: " + ", ".join(sig_missing_cols))
    n_sig = len(sig_rows)
    report.metrics["significant_degs"] = n_sig
    if not 2_000 <= n_sig <= 7_000:
        report.fail(f"significant DEGs = {n_sig}, expected 2000-7000")
    report.failures.extend(_validate_significant_rows(sig_rows, sig_cols, deg_rows, deg_cols))

    benchmark_failures, benchmark_values = _validate_benchmark_summary(
        outputs["benchmark_summary.csv"]
    )
    report.failures.extend(benchmark_failures)
    report.metrics["benchmark_summary"] = benchmark_values

    summary_text = _decode(outputs["sample_summary.txt"], "sample_summary.txt")
    tumor_count, normal_count = _parse_sample_counts(summary_text)
    report.metrics["tumor_samples"] = tumor_count
    report.metrics["normal_samples"] = normal_count
    if tumor_count is None or not 900 <= tumor_count <= 1150:
        report.fail(f"tumor sample count {tumor_count}, expected 900-1150")
    if normal_count is None or not 80 <= normal_count <= 150:
        report.fail(f"normal sample count {normal_count}, expected 80-150")

    try:
        manifest = json.loads(_decode(outputs["run_manifest.json"], "run_manifest.json"))
    except Exception as exc:
        report.fail(f"run_manifest.json parse failed: {exc}")
        manifest = {}
    manifest_checks = {
        "tool_name": lambda value: bool(str(value).strip()),
        "dataset": lambda value: "tcga" in str(value).lower() and "brca" in str(value).lower(),
        "comparison": lambda value: "tumor" in str(value).lower() and "normal" in str(value).lower(),
        "run_status": lambda value: str(value).lower() == "completed",
        "total_genes_tested": lambda value: isinstance(value, (int, float)) and value >= 18_000,
        "significant_degs": lambda value: isinstance(value, (int, float)) and 2_000 <= value <= 7_000,
        "tumor_samples": lambda value: isinstance(value, (int, float)) and 900 <= value <= 1150,
        "normal_samples": lambda value: isinstance(value, (int, float)) and 80 <= value <= 150,
        "fc_threshold": lambda value: value is not None,
        "padj_threshold": lambda value: value is not None,
    }
    for field_name, check in manifest_checks.items():
        if field_name not in manifest or not check(manifest[field_name]):
            report.fail(f"run_manifest.json field {field_name!r} is missing or invalid")

    if {"gene", "log2FC"}.issubset(sig_cols):
        truth = _truth_by_direction(reference_truth_csv)
        report.metrics["truth_genes_up"] = len(truth["up"])
        report.metrics["truth_genes_down"] = len(truth["down"])
        truth_scores = _truth_metrics(sig_rows, sig_cols, truth)
        report.metrics.update(truth_scores)
        for metric_name, threshold in PASS_THRESHOLDS.items():
            if truth_scores.get(metric_name, 0.0) < threshold:
                report.fail(
                    f"{metric_name} = {truth_scores.get(metric_name, 0.0):.4f}, "
                    f"expected >= {threshold:.2f}"
                )
        if gold_deg_results_all_csv and {"gene", "log2FC"}.issubset(deg_cols):
            gold_rows, gold_fields = _read_csv(
                gold_deg_results_all_csv, "reference/gold_output/deg_results_all.csv"
            )
            gold_cols = _column_map(gold_fields)
            checked, matched, max_delta = _gold_sanity(deg_rows, deg_cols, gold_rows, gold_cols, truth)
            report.metrics["gold_log2fc_anchors_checked"] = checked
            report.metrics["gold_log2fc_anchors_matched"] = matched
            report.metrics["gold_log2fc_max_abs_delta"] = max_delta
            if checked < 80:
                report.fail(f"only {checked} hidden gold sanity anchors available, expected >= 80")
            elif matched / checked < 0.90:
                report.fail(
                    f"hidden gold log2FC sanity matched {matched}/{checked} anchors, expected >= 90%"
                )

    return report.finish()
