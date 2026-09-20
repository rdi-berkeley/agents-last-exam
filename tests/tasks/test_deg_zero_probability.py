"""Synthetic numerical controls for zero and missing DEG probabilities."""

from __future__ import annotations

import csv
import io
import math

import pytest

from tasks.life_sciences.gene_expression_differential_analysis_functional_enrichment_analysis_1.scripts import (
    score_outputs as scorer,
)


PROBABILITY_CASES = [
    (0.0, 0.0, 0.0),
    (0.0, math.nextafter(0.0, 1.0), 0.0),
    (math.nextafter(0.0, 1.0), 0.0, 0.0),
    (-0.0, 1e-310, 0.0),
    (1e-310, -0.0, 0.0),
    (0.0, 1e-300, 0.0),
    (1e-300, 0.0, 0.0),
    (1e-310, 1e-320, 0.0),
    (1e-300, 1e-299, 1.0),
    (1e-299, 1e-300, 1.0),
    (0.0, 0.001, 297.0),
    (0.001, 0.0, 297.0),
    (0.001, 0.01, 1.0),
]
MISSING_VALUES = [None, "", "NA", "NaN", "None", "null"]


def tsv(columns, rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def deg_rows():
    return [
        {
            "id": name,
            "gene": name.upper(),
            "baseMean": 20,
            "log2FoldChange": fold_change,
            "lfcSE": 0.5,
            "stat": fold_change / 0.5,
            "pvalue": 0.0005,
            "padj": 0.001,
            "significant": label,
        }
        for name, fold_change, label in [
            ("up", 2, "upregulated"),
            ("down", -2, "downregulated"),
        ]
    ]


def score_rows(reference_rows, submitted_rows):
    enrichment = tsv(
        scorer.ENRICH_REQUIRED_COLUMNS,
        [
            {
                "Gene_set": "KEGG_2021_Human",
                "Term": "Synthetic pathway",
                "Overlap": "1/2",
                "P-value": 0.001,
                "Adjusted P-value": 0.01,
                "Old P-value": 0,
                "Old Adjusted P-value": 0,
                "Odds Ratio": 2,
                "Combined Score": 3,
                "Genes": "UP;DOWN",
            }
        ],
    )
    return scorer.score_submission(
        output_payloads=dict(
            zip(
                scorer.REQUIRED_FILES,
                [tsv(scorer.DEG_REQUIRED_COLUMNS, submitted_rows), enrichment, enrichment],
            )
        ),
        reference_deg_payload=tsv(scorer.DEG_REQUIRED_COLUMNS, reference_rows),
        reference_up_payload=enrichment,
        reference_down_payload=enrichment,
    )


@pytest.mark.parametrize("column", ["pvalue", "padj"])
@pytest.mark.parametrize("reference,submitted,error", PROBABILITY_CASES)
def test_probability_log_floor_and_score(column, reference, submitted, error):
    reference_rows, submitted_rows = deg_rows(), deg_rows()
    reference_rows[0][column] = reference
    submitted_rows[0][column] = submitted
    result = score_rows(reference_rows, submitted_rows)
    assert not result.failures, result.to_dict()
    assert result.metrics["deg_significant_gene_count"] == 2
    assert result.metrics[f"{column}_log10_mae_on_reference_significant_genes"] == error / 2
    column_score = max(0.0, 1.0 - error / 2)
    assert result.metrics["auxiliary_column_scores"][f"{column}_log10_score"] == column_score
    assert result.score == round(0.96 + 0.04 * column_score, 12)
    assert result.passed


@pytest.mark.parametrize("missing", MISSING_VALUES)
@pytest.mark.parametrize("reference,error", [(0.0, 300.0), (1e-310, 300.0), (0.001, 3.0)])
def test_missing_submitted_padj_still_means_one(missing, reference, error):
    reference_rows, submitted_rows = deg_rows(), deg_rows()
    reference_rows[0]["padj"] = reference
    submitted_rows[0].update(padj=missing, significant="no significant")
    result = score_rows(reference_rows, submitted_rows)
    assert not result.failures, result.to_dict()
    assert result.metrics["padj_log10_mae_on_reference_significant_genes"] == error / 2
    assert result.metrics["deg_up"]["fn"] == 1
    assert result.metrics["deg_significant_gene_count"] == 2
    assert result.metrics["auxiliary_effective_score"] == 0.5
    assert result.score == 0.675


@pytest.mark.parametrize("missing", MISSING_VALUES)
def test_missing_reference_padj_remains_excluded_from_numeric_comparison(missing):
    reference_rows = deg_rows()
    reference_rows.append(
        {
            **reference_rows[0],
            "id": "missing",
            "gene": "MISSING",
            "padj": missing,
            "significant": "no significant",
        }
    )
    submitted_rows = [row.copy() for row in reference_rows]
    baseline = score_rows(reference_rows, submitted_rows)
    assert not baseline.failures and baseline.score == 1
    submitted_rows[-1].update(padj=0.0, significant="upregulated")
    result = score_rows(reference_rows, submitted_rows)
    assert not result.failures, result.to_dict()
    assert result.metrics["deg_significant_gene_count"] == 2
    assert result.metrics["padj_log10_mae_on_reference_significant_genes"] == 0
    assert result.metrics["deg_up"]["fp"] == 1
    assert result.score < 1


@pytest.mark.parametrize("missing", MISSING_VALUES)
def test_missing_raw_pvalue_is_still_rejected(missing):
    reference_rows, submitted_rows = deg_rows(), deg_rows()
    submitted_rows[0]["pvalue"] = missing
    result = score_rows(reference_rows, submitted_rows)
    assert result.score == 0 and not result.passed
    assert any("non-numeric value in column pvalue" in failure for failure in result.failures)
