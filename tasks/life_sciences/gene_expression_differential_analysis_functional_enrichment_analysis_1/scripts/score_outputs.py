"""Structured scorer for the BRCA DEG plus KEGG enrichment task."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REQUIRED_FILES = [
    "BRCA_deseq2_results.tsv",
    "BRCA_upregulated_genes_kegg_enrichment.tsv",
    "BRCA_downregulated_genes_kegg_enrichment.tsv",
]

DEG_REQUIRED_COLUMNS = [
    "id",
    "baseMean",
    "log2FoldChange",
    "lfcSE",
    "stat",
    "pvalue",
    "padj",
    "gene",
    "significant",
]

ENRICH_REQUIRED_COLUMNS = [
    "Gene_set",
    "Term",
    "Overlap",
    "P-value",
    "Adjusted P-value",
    "Old P-value",
    "Old Adjusted P-value",
    "Odds Ratio",
    "Combined Score",
    "Genes",
]

ALLOWED_SIGNIFICANCE = {"upregulated", "downregulated", "no significant"}
EXPECTED_LIBRARY = "kegg_2021_human"
TOP_REFERENCE_TERMS = 20
TOP_PREDICTED_TERMS = 50
LFC_MAE_TOLERANCE = 0.5
LFCSE_MAE_TOLERANCE = 0.25
STAT_MAE_TOLERANCE = 1.0
BASEMEAN_LOG1P_MAE_TOLERANCE = 0.25
PVALUE_LOG10_MAE_TOLERANCE = 1.0
PADJ_LOG10_MAE_TOLERANCE = 1.0
PASS_THRESHOLD = 0.85


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
        self.score = 0.0 if self.failures else float(self.score)
        self.passed = not self.failures and self.score >= PASS_THRESHOLD
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "passed": self.passed,
            "failures": self.failures,
            "warnings": self.warnings,
            "metrics": self.metrics,
        }


@dataclass
class DegTable:
    rows_by_id: dict[str, dict[str, Any]]
    labels: dict[str, str]
    log2fc: dict[str, float]


@dataclass
class EnrichmentTable:
    rows: list[dict[str, str]]


def _decode(payload: bytes, filename: str) -> str:
    try:
        return payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{filename} is not valid UTF-8 text") from exc


def _read_tsv(payload: bytes, filename: str) -> tuple[list[dict[str, str]], list[str]]:
    text = _decode(payload, filename)
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    rows = list(reader)
    return rows, list(reader.fieldnames or [])


def _float(value: str | None, *, allow_zero: bool = True) -> float | None:
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
    if not allow_zero and parsed == 0:
        return None
    return parsed


def _normalize_term(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).lower()


def _expected_significance(log2fc: float, padj: float) -> str:
    if padj is None:
        return "no significant"
    if log2fc > 1.0 and padj < 0.05:
        return "upregulated"
    if log2fc < -1.0 and padj < 0.05:
        return "downregulated"
    return "no significant"


def _load_reference_gene_map(reference_deg_payload: bytes) -> dict[str, str]:
    rows, fields = _read_tsv(reference_deg_payload, "reference/BRCA_deseq2_results.tsv")
    missing_columns = [column for column in ["id", "gene"] if column not in fields]
    if missing_columns:
        raise ValueError(
            "reference/BRCA_deseq2_results.tsv must include id and gene columns to define evaluator-owned symbol truth"
        )
    mapping: dict[str, str] = {}
    for row in rows:
        gene_id = (row.get("id") or "").strip()
        gene_symbol = (row.get("gene") or "").strip()
        if not gene_id or not gene_symbol:
            raise ValueError("reference/BRCA_deseq2_results.tsv contains blank gene ids or symbols")
        if gene_id in mapping and mapping[gene_id] != gene_symbol:
            raise ValueError(f"reference/BRCA_deseq2_results.tsv contains conflicting symbols for {gene_id}")
        mapping[gene_id] = gene_symbol
    if not mapping:
        raise ValueError("reference/BRCA_deseq2_results.tsv is empty")
    return mapping


def _validate_deg_table(
    payload: bytes,
    *,
    filename: str,
    expected_gene_map: dict[str, str],
    report: ScoreReport,
) -> DegTable | None:
    rows, fields = _read_tsv(payload, filename)
    missing_columns = [column for column in DEG_REQUIRED_COLUMNS if column not in fields]
    if missing_columns:
        report.fail(f"{filename} is missing required columns: {missing_columns}")
        return None
    if not rows:
        report.fail(f"{filename} is empty")
        return None

    rows_by_id: dict[str, dict[str, Any]] = {}
    labels: dict[str, str] = {}
    log2fc: dict[str, float] = {}
    label_mismatches = 0
    symbol_mismatches = 0

    for row in rows:
        gene_id = (row.get("id") or "").strip()
        if not gene_id:
            report.fail(f"{filename} contains a blank gene id")
            return None
        if gene_id in rows_by_id:
            report.fail(f"{filename} contains duplicate gene id {gene_id}")
            return None
        expected_symbol = expected_gene_map.get(gene_id)
        if expected_symbol is None:
            report.fail(f"{filename} contains unexpected gene id {gene_id}")
            return None

        numeric_fields = {}
        for column in ["baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"]:
            value = _float(row.get(column))
            if value is None and column != "padj":
                report.fail(f"{filename} has a non-numeric value in column {column} for {gene_id}")
                return None
            if value is not None:
                if column in {"baseMean", "lfcSE"} and value < 0:
                    report.fail(f"{filename} has an invalid negative value in column {column} for {gene_id}")
                    return None
                if column in {"pvalue", "padj"} and not (0.0 <= value <= 1.0):
                    report.fail(f"{filename} has an out-of-range probability in column {column} for {gene_id}")
                    return None
            numeric_fields[column] = value

        observed_symbol = (row.get("gene") or "").strip()
        if observed_symbol != expected_symbol:
            symbol_mismatches += 1

        observed_label = (row.get("significant") or "").strip().lower()
        if observed_label not in ALLOWED_SIGNIFICANCE:
            report.fail(f"{filename} has invalid significance label {observed_label!r} for {gene_id}")
            return None
        expected_label = _expected_significance(
            numeric_fields["log2FoldChange"],
            numeric_fields["padj"],
        )
        if observed_label != expected_label:
            label_mismatches += 1

        rows_by_id[gene_id] = {
            **row,
            **numeric_fields,
            "gene": observed_symbol,
            "significant": observed_label,
        }
        labels[gene_id] = observed_label
        log2fc[gene_id] = numeric_fields["log2FoldChange"]

    if len(rows_by_id) != len(expected_gene_map):
        report.fail(
            f"{filename} has {len(rows_by_id)} rows but expected {len(expected_gene_map)} from hidden-reference symbol truth"
        )
        return None

    missing_ids = sorted(set(expected_gene_map) - set(rows_by_id))
    if missing_ids:
        report.fail(f"{filename} is missing gene ids such as {missing_ids[0]}")
        return None

    if symbol_mismatches:
        report.fail(
            f"{filename} has {symbol_mismatches} gene-symbol mismatches against evaluator-owned symbol truth"
        )
        return None
    if label_mismatches:
        report.fail(
            f"{filename} has {label_mismatches} rows where `significant` disagrees with log2FoldChange/padj"
        )
        return None

    return DegTable(rows_by_id=rows_by_id, labels=labels, log2fc=log2fc)


def _validate_enrichment_table(payload: bytes, *, filename: str, report: ScoreReport) -> EnrichmentTable | None:
    rows, fields = _read_tsv(payload, filename)
    missing_columns = [column for column in ENRICH_REQUIRED_COLUMNS if column not in fields]
    if missing_columns:
        report.fail(f"{filename} is missing required columns: {missing_columns}")
        return None
    if not rows:
        report.fail(f"{filename} is empty")
        return None

    for index, row in enumerate(rows, start=1):
        gene_set = _normalize_term(row.get("Gene_set") or "")
        if gene_set != EXPECTED_LIBRARY:
            report.fail(f"{filename} row {index} has unexpected Gene_set {row.get('Gene_set')!r}")
            return None
        term = (row.get("Term") or "").strip()
        if not term:
            report.fail(f"{filename} row {index} has a blank Term")
            return None
        for column in ["P-value", "Adjusted P-value", "Old P-value", "Old Adjusted P-value", "Odds Ratio", "Combined Score"]:
            value = _float(row.get(column))
            if value is None:
                report.fail(f"{filename} row {index} has a non-numeric value in column {column}")
                return None
            if column in {"P-value", "Adjusted P-value", "Old P-value", "Old Adjusted P-value"} and not (0.0 <= value <= 1.0):
                report.fail(f"{filename} row {index} has an out-of-range probability in column {column}")
                return None
            if column == "Odds Ratio" and value <= 0:
                report.fail(f"{filename} row {index} has a non-positive Odds Ratio")
                return None
            if column == "Combined Score" and value < 0:
                report.fail(f"{filename} row {index} has a negative Combined Score")
                return None
        if not (row.get("Overlap") or "").strip():
            report.fail(f"{filename} row {index} has a blank Overlap value")
            return None
        if not (row.get("Genes") or "").strip():
            report.fail(f"{filename} row {index} has a blank Genes value")
            return None

    return EnrichmentTable(rows=rows)


def _f1_for_label(truth: dict[str, str], pred: dict[str, str], label: str) -> dict[str, float]:
    tp = sum(1 for gene_id, truth_label in truth.items() if truth_label == label and pred.get(gene_id) == label)
    fp = sum(1 for gene_id, pred_label in pred.items() if pred_label == label and truth.get(gene_id) != label)
    fn = sum(1 for gene_id, truth_label in truth.items() if truth_label == label and pred.get(gene_id) != label)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _enrichment_term_recall(reference_rows: list[dict[str, str]], predicted_rows: list[dict[str, str]]) -> dict[str, Any]:
    target_terms: list[str] = []
    for row in reference_rows:
        adj_p = _float(row.get("Adjusted P-value"))
        if adj_p is not None and adj_p < 0.05:
            target_terms.append(_normalize_term(row["Term"]))
        if len(target_terms) >= TOP_REFERENCE_TERMS:
            break
    if not target_terms:
        target_terms = [_normalize_term(row["Term"]) for row in reference_rows[: min(10, len(reference_rows))]]
    target_set = set(target_terms)
    candidate_terms = {
        _normalize_term(row["Term"]) for row in predicted_rows[: max(TOP_PREDICTED_TERMS, len(target_terms))]
    }
    overlap = len(target_set & candidate_terms)
    recall = overlap / len(target_set) if target_set else 0.0
    return {
        "target_terms": len(target_set),
        "candidate_terms": len(candidate_terms),
        "overlap": overlap,
        "recall": recall,
    }


def _score_from_mae(mae: float, tolerance: float) -> float:
    return max(0.0, 1.0 - (mae / tolerance))


def score_submission(
    *,
    output_payloads: dict[str, bytes],
    reference_deg_payload: bytes,
    reference_up_payload: bytes,
    reference_down_payload: bytes,
) -> ScoreReport:
    report = ScoreReport()
    expected_gene_map = _load_reference_gene_map(reference_deg_payload)

    missing_payloads = [name for name in REQUIRED_FILES if name not in output_payloads]
    if missing_payloads:
        report.fail(f"missing required output payloads: {missing_payloads}")
        return report.finish()

    submitted_deg = _validate_deg_table(
        output_payloads["BRCA_deseq2_results.tsv"],
        filename="BRCA_deseq2_results.tsv",
        expected_gene_map=expected_gene_map,
        report=report,
    )
    reference_deg = _validate_deg_table(
        reference_deg_payload,
        filename="reference/BRCA_deseq2_results.tsv",
        expected_gene_map=expected_gene_map,
        report=report,
    )
    submitted_up = _validate_enrichment_table(
        output_payloads["BRCA_upregulated_genes_kegg_enrichment.tsv"],
        filename="BRCA_upregulated_genes_kegg_enrichment.tsv",
        report=report,
    )
    submitted_down = _validate_enrichment_table(
        output_payloads["BRCA_downregulated_genes_kegg_enrichment.tsv"],
        filename="BRCA_downregulated_genes_kegg_enrichment.tsv",
        report=report,
    )
    reference_up = _validate_enrichment_table(
        reference_up_payload,
        filename="reference/BRCA_upregulated_genes_kegg_enrichment.tsv",
        report=report,
    )
    reference_down = _validate_enrichment_table(
        reference_down_payload,
        filename="reference/BRCA_downregulated_genes_kegg_enrichment.tsv",
        report=report,
    )
    if report.failures:
        return report.finish()

    assert submitted_deg is not None
    assert reference_deg is not None
    assert submitted_up is not None
    assert submitted_down is not None
    assert reference_up is not None
    assert reference_down is not None

    f1_up = _f1_for_label(reference_deg.labels, submitted_deg.labels, "upregulated")
    f1_down = _f1_for_label(reference_deg.labels, submitted_deg.labels, "downregulated")
    deg_direction_f1 = (f1_up["f1"] + f1_down["f1"]) / 2.0

    significant_ids = [
        gene_id for gene_id, label in reference_deg.labels.items() if label in {"upregulated", "downregulated"}
    ]
    lfc_mae = sum(
        abs(submitted_deg.log2fc[gene_id] - reference_deg.log2fc[gene_id]) for gene_id in significant_ids
    ) / len(significant_ids)
    lfc_score = _score_from_mae(lfc_mae, LFC_MAE_TOLERANCE)

    basemean_log1p_mae = sum(
        abs(
            math.log1p(submitted_deg.rows_by_id[gene_id]["baseMean"])
            - math.log1p(reference_deg.rows_by_id[gene_id]["baseMean"])
        )
        for gene_id in significant_ids
    ) / len(significant_ids)
    lfcse_mae = sum(
        abs(submitted_deg.rows_by_id[gene_id]["lfcSE"] - reference_deg.rows_by_id[gene_id]["lfcSE"])
        for gene_id in significant_ids
    ) / len(significant_ids)
    stat_mae = sum(
        abs(submitted_deg.rows_by_id[gene_id]["stat"] - reference_deg.rows_by_id[gene_id]["stat"])
        for gene_id in significant_ids
    ) / len(significant_ids)
    pvalue_log10_mae = sum(
        abs(
            math.log10(max(submitted_deg.rows_by_id[gene_id]["pvalue"], 1e-300))
            - math.log10(max(reference_deg.rows_by_id[gene_id]["pvalue"], 1e-300))
        )
        for gene_id in significant_ids
    ) / len(significant_ids)
    padj_log10_mae = sum(
        abs(
            math.log10(max(submitted_deg.rows_by_id[gene_id]["padj"] or 1.0, 1e-300))
            - math.log10(max(reference_deg.rows_by_id[gene_id]["padj"] or 1.0, 1e-300))
        )
        for gene_id in significant_ids
    ) / len(significant_ids)

    auxiliary_column_scores = {
        "baseMean_log1p_score": _score_from_mae(basemean_log1p_mae, BASEMEAN_LOG1P_MAE_TOLERANCE),
        "lfcSE_score": _score_from_mae(lfcse_mae, LFCSE_MAE_TOLERANCE),
        "stat_score": _score_from_mae(stat_mae, STAT_MAE_TOLERANCE),
        "pvalue_log10_score": _score_from_mae(pvalue_log10_mae, PVALUE_LOG10_MAE_TOLERANCE),
        "padj_log10_score": _score_from_mae(padj_log10_mae, PADJ_LOG10_MAE_TOLERANCE),
    }
    auxiliary_numeric_score = sum(auxiliary_column_scores.values()) / len(auxiliary_column_scores)
    auxiliary_effective_score = min(auxiliary_numeric_score, deg_direction_f1, lfc_score)

    up_enrichment = _enrichment_term_recall(reference_up.rows, submitted_up.rows)
    down_enrichment = _enrichment_term_recall(reference_down.rows, submitted_down.rows)
    enrichment_score = (up_enrichment["recall"] + down_enrichment["recall"]) / 2.0

    raw_score = (
        0.45 * deg_direction_f1
        + 0.20 * lfc_score
        + 0.20 * auxiliary_effective_score
        + 0.15 * enrichment_score
    )
    if raw_score < 0.05:
        raw_score = 0.0

    report.score = round(min(max(raw_score, 0.0), 1.0), 12)
    report.metrics.update(
        {
            "deg_direction_f1": deg_direction_f1,
            "deg_up": f1_up,
            "deg_down": f1_down,
            "deg_significant_gene_count": len(significant_ids),
            "log2fc_mae_on_reference_significant_genes": lfc_mae,
            "log2fc_score": lfc_score,
            "baseMean_log1p_mae_on_reference_significant_genes": basemean_log1p_mae,
            "lfcSE_mae_on_reference_significant_genes": lfcse_mae,
            "stat_mae_on_reference_significant_genes": stat_mae,
            "pvalue_log10_mae_on_reference_significant_genes": pvalue_log10_mae,
            "padj_log10_mae_on_reference_significant_genes": padj_log10_mae,
            "auxiliary_column_scores": auxiliary_column_scores,
            "auxiliary_numeric_score": auxiliary_numeric_score,
            "auxiliary_effective_score": auxiliary_effective_score,
            "up_enrichment": up_enrichment,
            "down_enrichment": down_enrichment,
            "enrichment_score": enrichment_score,
            "pass_threshold": PASS_THRESHOLD,
        }
    )
    if report.score < PASS_THRESHOLD:
        report.warn("structured comparison score is below the benchmark pass threshold")
    return report.finish()


def _read_path(path: Path) -> bytes:
    return path.read_bytes()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    args = parser.parse_args()

    output_payloads = {
        name: _read_path(args.output_dir / name)
        for name in REQUIRED_FILES
    }
    report = score_submission(
        output_payloads=output_payloads,
        reference_deg_payload=_read_path(args.reference_dir / "BRCA_deseq2_results.tsv"),
        reference_up_payload=_read_path(args.reference_dir / "BRCA_upregulated_genes_kegg_enrichment.tsv"),
        reference_down_payload=_read_path(args.reference_dir / "BRCA_downregulated_genes_kegg_enrichment.tsv"),
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
