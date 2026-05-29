"""Deterministic scorer for the TCGA-BRCA ER-status DGE task."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any

REQUIRED_FILES = [
    "deg_table.csv",
    "top_genes.json",
    "volcano_plot.png",
    "analysis.py",
    "summary.json",
]

DEG_REQUIRED_COLUMNS = [
    "gene_id",
    "gene_symbol",
    "log2_fold_change",
    "p_value",
    "adjusted_p_value",
]

REFERENCE_TOP_N = 50
PASS_TOP50_OVERLAP = 45
MIN_DEG_ROWS = 500
EXPECTED_SIGNIFICANT_COUNT = 533


@dataclass
class ScoreReport:
    score: float
    passed: bool
    reasons: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "passed": self.passed,
            "reasons": self.reasons,
            "details": self.details,
        }


def _decode(payload: bytes, label: str) -> str:
    try:
        return payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not valid UTF-8 text") from exc


def _load_json(payload: bytes, label: str) -> Any:
    try:
        return json.loads(_decode(payload, label))
    except Exception as exc:
        raise ValueError(f"{label} is not valid JSON") from exc


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "nan", "na", "none", "null", "inf", "-inf"}:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _as_int(value: Any) -> int | None:
    parsed = _as_float(value)
    if parsed is None or abs(parsed - round(parsed)) > 1e-9:
        return None
    return int(round(parsed))


def _load_deg_rows(payload: bytes) -> tuple[list[dict[str, str]], list[str]]:
    text = _decode(payload, "deg_table.csv")
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    return rows, list(reader.fieldnames or [])


def _load_reference_top50(reference_top50_json: bytes) -> set[str]:
    data = _load_json(reference_top50_json, "reference_top50_degs.json")
    symbols = data.get("top_50_gene_symbols")
    if not isinstance(symbols, list) or len(symbols) != REFERENCE_TOP_N:
        raise ValueError("reference_top50_degs.json must contain 50 top gene symbols")
    return {str(symbol) for symbol in symbols}


def _load_reference_summary(reference_summary_json: bytes, reference_top50_json: bytes) -> dict[str, int]:
    summary = _load_json(reference_summary_json, "reference_summary.json")
    top50_meta = _load_json(reference_top50_json, "reference_top50_degs.json")
    n_significant = top50_meta.get("n_significant")
    if n_significant is None:
        n_significant = int(summary.get("n_significant_up", 0)) + int(summary.get("n_significant_down", 0))
    return {
        "n_er_positive": int(summary["n_er_positive"]),
        "n_er_negative": int(summary["n_er_negative"]),
        "n_genes_tested": int(summary["n_genes_tested"]),
        "n_significant": int(n_significant),
    }


def _rank_top_symbols(rows: list[dict[str, str]]) -> tuple[list[str], list[str]]:
    parsed: list[tuple[float, float, str, str]] = []
    issues: list[str] = []
    seen_symbols: set[str] = set()
    for index, row in enumerate(rows, start=2):
        symbol = (row.get("gene_symbol") or "").strip()
        padj = _as_float(row.get("adjusted_p_value"))
        log2fc = _as_float(row.get("log2_fold_change"))
        pval = _as_float(row.get("p_value"))
        if not symbol:
            issues.append(f"blank gene_symbol at CSV row {index}")
            continue
        if symbol in seen_symbols:
            issues.append(f"duplicate gene_symbol {symbol}")
        seen_symbols.add(symbol)
        if padj is None or log2fc is None or pval is None:
            issues.append(f"non-numeric ranking/stat column at CSV row {index}")
            continue
        if not (0.0 <= padj <= 1.0) or not (0.0 <= pval <= 1.0):
            issues.append(f"p-value out of range at CSV row {index}")
            continue
        parsed.append((padj, -abs(log2fc), symbol, symbol))
    parsed.sort()
    return [item[3] for item in parsed[:REFERENCE_TOP_N]], issues


def _count_threshold_significant(rows: list[dict[str, str]]) -> tuple[int, int]:
    threshold_count = 0
    mismatches = 0
    for row in rows:
        log2fc = _as_float(row.get("log2_fold_change"))
        padj = _as_float(row.get("adjusted_p_value"))
        if log2fc is None or padj is None:
            continue
        expected = abs(log2fc) > 1.5 and padj < 0.01
        observed_raw = str(row.get("significant", "")).strip().lower()
        if expected:
            threshold_count += 1
        if observed_raw in {"true", "false", "1", "0", "yes", "no"}:
            observed = observed_raw in {"true", "1", "yes"}
            if observed != expected:
                mismatches += 1
    return threshold_count, mismatches


def _extract_top_genes(data: Any) -> list[str]:
    if isinstance(data, list):
        return [str(item.get("gene_symbol", item)) if isinstance(item, dict) else str(item) for item in data]
    if not isinstance(data, dict):
        return []
    for key in ("top_50_gene_symbols", "top_genes", "genes"):
        value = data.get(key)
        if isinstance(value, list):
            return [str(item.get("gene_symbol", item)) if isinstance(item, dict) else str(item) for item in value]
    value = data.get("top_50_degs")
    if isinstance(value, list):
        return [str(item.get("gene_symbol", "")) for item in value if isinstance(item, dict)]
    return []


def _validate_png(payload: bytes) -> tuple[bool, dict[str, Any], str | None]:
    if len(payload) < 1024:
        return False, {"png_size_bytes": len(payload)}, "volcano_plot.png is too small"
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return False, {"png_size_bytes": len(payload)}, "volcano_plot.png has invalid PNG signature"
    offset = 8
    width = height = None
    saw_iend = False
    try:
        while offset + 12 <= len(payload):
            length = struct.unpack(">I", payload[offset : offset + 4])[0]
            kind = payload[offset + 4 : offset + 8]
            data_start = offset + 8
            data_end = data_start + length
            crc_end = data_end + 4
            if crc_end > len(payload):
                return False, {}, "volcano_plot.png has truncated PNG chunk"
            expected_crc = struct.unpack(">I", payload[data_end:crc_end])[0]
            actual_crc = zlib.crc32(kind + payload[data_start:data_end]) & 0xFFFFFFFF
            if expected_crc != actual_crc:
                return False, {}, f"volcano_plot.png has bad CRC for chunk {kind.decode(errors='replace')}"
            if kind == b"IHDR":
                width, height = struct.unpack(">II", payload[data_start : data_start + 8])
            if kind == b"IEND":
                saw_iend = True
                break
            offset = crc_end
    except Exception as exc:
        return False, {}, f"volcano_plot.png parse failed: {exc}"
    details = {"png_size_bytes": len(payload), "png_width": width, "png_height": height}
    if not saw_iend:
        return False, details, "volcano_plot.png missing IEND chunk"
    if not width or not height or width < 300 or height < 200:
        return False, details, "volcano_plot.png dimensions are too small"
    return True, details, None


def score_submission(
    outputs: dict[str, bytes],
    *,
    reference_top50_json: bytes,
    reference_summary_json: bytes,
) -> ScoreReport:
    reasons: list[str] = []
    details: dict[str, Any] = {}
    missing = [name for name in REQUIRED_FILES if name not in outputs or not outputs[name]]
    if missing:
        return ScoreReport(0.0, False, [f"missing or empty required files: {missing}"], details)

    try:
        reference_top50 = _load_reference_top50(reference_top50_json)
        reference_summary = _load_reference_summary(reference_summary_json, reference_top50_json)
        rows, fields = _load_deg_rows(outputs["deg_table.csv"])
        top_genes_data = _load_json(outputs["top_genes.json"], "top_genes.json")
        summary_data = _load_json(outputs["summary.json"], "summary.json")
        analysis_text = _decode(outputs["analysis.py"], "analysis.py")
    except Exception as exc:
        return ScoreReport(0.0, False, [str(exc)], details)

    score = 0.0

    missing_columns = [column for column in DEG_REQUIRED_COLUMNS if column not in fields]
    if missing_columns:
        reasons.append(f"deg_table.csv missing columns: {missing_columns}")
    elif rows:
        score += 0.10
    else:
        reasons.append("deg_table.csv has no rows")

    png_ok, png_details, png_reason = _validate_png(outputs["volcano_plot.png"])
    details.update(png_details)
    if png_ok:
        score += 0.05
    elif png_reason:
        reasons.append(png_reason)

    if len(analysis_text.strip()) >= 80 and ("mann" in analysis_text.lower() or "wilcoxon" in analysis_text.lower()):
        score += 0.05
    else:
        reasons.append("analysis.py is missing or does not describe the Mann-Whitney/Wilcoxon workflow")

    top_from_table, table_issues = _rank_top_symbols(rows)
    details["deg_rows"] = len(rows)
    if table_issues:
        details["table_issues_sample"] = table_issues[:5]
    overlap = len(reference_top50 & set(top_from_table))
    details["top50_overlap_from_deg_table"] = overlap
    if len(rows) >= MIN_DEG_ROWS:
        score += 0.50 * min(overlap / PASS_TOP50_OVERLAP, 1.0)
    else:
        reasons.append(f"deg_table.csv has {len(rows)} rows; expected at least {MIN_DEG_ROWS}")

    top_genes = _extract_top_genes(top_genes_data)
    top_json_overlap = len(reference_top50 & set(top_genes[:REFERENCE_TOP_N]))
    details["top50_overlap_from_top_genes_json"] = top_json_overlap
    score += 0.10 * min(top_json_overlap / PASS_TOP50_OVERLAP, 1.0)

    threshold_count, label_mismatches = _count_threshold_significant(rows)
    details["threshold_significant_count"] = threshold_count
    details["significant_label_mismatches"] = label_mismatches
    if rows and abs(threshold_count - EXPECTED_SIGNIFICANT_COUNT) <= 5 and label_mismatches == 0:
        score += 0.10
    else:
        reasons.append("significant DEG count or labels do not match the deterministic threshold")

    summary_expected_matches = 0
    summary_keys = ["n_er_positive", "n_er_negative", "n_genes_tested", "n_significant"]
    for key in summary_keys:
        if key == "n_significant" and summary_data.get(key) is None:
            up = _as_int(summary_data.get("n_significant_up"))
            down = _as_int(summary_data.get("n_significant_down"))
            observed = None if up is None or down is None else up + down
        else:
            observed = _as_int(summary_data.get(key))
        expected = reference_summary[key]
        if observed is not None and abs(observed - expected) <= (5 if key == "n_significant" else 0):
            summary_expected_matches += 1
    details["summary_keys_matched"] = summary_expected_matches
    score += 0.10 * (summary_expected_matches / len(summary_keys))

    score = round(min(score, 1.0), 6)
    passed = score >= 0.95 and overlap >= PASS_TOP50_OVERLAP and len(rows) >= MIN_DEG_ROWS
    return ScoreReport(score, passed, reasons, details)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a local TCGA-BRCA ER DGE output directory.")
    parser.add_argument("output_dir")
    parser.add_argument("--reference-dir", required=True)
    args = parser.parse_args()

    from pathlib import Path

    output_dir = Path(args.output_dir)
    reference_dir = Path(args.reference_dir)
    outputs = {name: (output_dir / name).read_bytes() for name in REQUIRED_FILES if (output_dir / name).exists()}
    report = score_submission(
        outputs,
        reference_top50_json=(reference_dir / "reference_top50_degs.json").read_bytes(),
        reference_summary_json=(reference_dir / "reference_summary.json").read_bytes(),
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
