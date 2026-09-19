"""Preserve the scientific DEG contract while estimator reproducibility is audited."""

from __future__ import annotations

import csv
import io
import math
import os

import pytest

from tasks.life_sciences.gene_expression_differential_analysis_functional_enrichment_analysis_1.scripts import (
    score_outputs as scorer,
)


def tsv(columns, rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


@pytest.fixture
def tables():
    rows = []
    for gene, lfc, padj in [("up", 2, 0.001), ("down", -2, 0.001), ("null", 0.2, 0.8)]:
        rows.append(
            {
                "id": gene,
                "gene": gene.upper(),
                "baseMean": 20,
                "log2FoldChange": lfc,
                "lfcSE": 0.5,
                "stat": lfc / 0.5,
                "pvalue": padj / 2,
                "padj": padj,
                "significant": scorer._expected_significance(lfc, padj),
            }
        )
    enrichment = [
        {
            "Gene_set": "KEGG_2021_Human",
            "Term": "Example pathway",
            "Overlap": "1/2",
            "P-value": 0.001,
            "Adjusted P-value": 0.01,
            "Old P-value": 0,
            "Old Adjusted P-value": 0,
            "Odds Ratio": 2,
            "Combined Score": 3,
            "Genes": "UP",
        }
    ]
    return rows, enrichment


def score(rows, enrichment, reference_rows, reference_enrichment):
    deg = tsv(scorer.DEG_REQUIRED_COLUMNS, rows)
    enrich = tsv(scorer.ENRICH_REQUIRED_COLUMNS, enrichment)
    reference = tsv(scorer.ENRICH_REQUIRED_COLUMNS, reference_enrichment)
    return scorer.score_submission(
        output_payloads=dict(zip(scorer.REQUIRED_FILES, [deg, enrich, enrich])),
        reference_deg_payload=tsv(scorer.DEG_REQUIRED_COLUMNS, reference_rows),
        reference_up_payload=reference,
        reference_down_payload=reference,
    )


@pytest.mark.parametrize("encoding", ["unchanged", "scientific", "reordered", "mixed-label-case"])
def test_exact_numeric_and_structural_equivalences(tables, encoding):
    reference, enrichment = tables
    rows = [row.copy() for row in reference]
    if encoding == "scientific":
        for row in rows:
            for column in ["baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"]:
                row[column] = format(row[column], ".17e")
    elif encoding == "reordered":
        rows.reverse()
    elif encoding == "mixed-label-case":
        for row in rows:
            row["significant"] = row["significant"].upper()
    result = score(rows, enrichment, reference, enrichment)
    assert result.score == 1 and result.passed, result.to_dict()


@pytest.mark.parametrize(
    "lfc,padj,expected",
    [
        (1, 0.001, "no significant"),
        (-1, 0.001, "no significant"),
        (math.nextafter(1, math.inf), 0.001, "upregulated"),
        (math.nextafter(-1, -math.inf), 0.001, "downregulated"),
        (2, 0.05, "no significant"),
        (-2, 0.05, "no significant"),
        (2, math.nextafter(0.05, 0), "upregulated"),
        (-2, math.nextafter(0.05, 0), "downregulated"),
        (2, math.nextafter(0.05, 1), "no significant"),
        (2, None, "no significant"),
    ],
)
def test_scientific_cutoffs_remain_strict(lfc, padj, expected):
    assert scorer._expected_significance(lfc, padj) == expected


@pytest.mark.parametrize(
    "column,value",
    [
        ("id", "unknown"),
        ("gene", "WRONG_SYMBOL"),
        ("significant", "no significant"),
        ("log2FoldChange", "invalid"),
        ("lfcSE", -0.1),
        ("baseMean", -1),
        ("stat", "NaN"),
        ("pvalue", "inf"),
        ("pvalue", 1.01),
        ("padj", -0.01),
    ],
)
def test_malformed_or_inconsistent_submissions_fail(tables, column, value):
    reference, enrichment = tables
    rows = [row.copy() for row in reference]
    rows[0][column] = value
    result = score(rows, enrichment, reference, enrichment)
    assert result.score == 0 and result.failures


@pytest.mark.parametrize("change", ["missing", "duplicate"])
def test_gene_coverage_contract_is_preserved(tables, change):
    reference, enrichment = tables
    rows = reference[:-1] if change == "missing" else reference + [reference[0]]
    assert score(rows, enrichment, reference, enrichment).score == 0


def test_reversed_contrast_is_not_a_numeric_equivalence(tables):
    reference, enrichment = tables
    rows = [row.copy() for row in reference]
    for row in rows:
        row["log2FoldChange"] *= -1
        row["stat"] *= -1
        row["significant"] = scorer._expected_significance(row["log2FoldChange"], row["padj"])
    result = score(rows, enrichment, reference, enrichment)
    assert result.score == 0.15 and not result.passed
    assert result.metrics["deg_direction_f1"] == 0


def test_boundary_change_and_substantive_change_are_not_silently_conflated(tables):
    reference, enrichment = tables
    reference = [row.copy() for row in reference]
    reference[0]["padj"] = 0.04996149826013284
    near = [row.copy() for row in reference]
    far = [row.copy() for row in reference]
    near[0]["padj"] = 0.05001405396800041
    far[0]["padj"] = 0.5
    for rows in (near, far):
        rows[0]["significant"] = "no significant"
    near_result = score(near, enrichment, reference, enrichment)
    far_result = score(far, enrichment, reference, enrichment)
    assert near_result.score < 1 and far_result.score < 1
    assert (
        near_result.metrics["padj_log10_mae_on_reference_significant_genes"]
        < far_result.metrics["padj_log10_mae_on_reference_significant_genes"]
    )


def test_current_numeric_score_has_no_unmeasured_full_credit_epsilon():
    assert scorer._score_from_mae(0, 0.5) == 1
    assert scorer._score_from_mae(0.0001, 0.5) < 1
    assert scorer._score_from_mae(0.25, 0.5) == 0.5
    assert scorer._score_from_mae(0.5, 0.5) == 0
    assert scorer.PASS_THRESHOLD == 0.85


def test_wrong_enrichment_library_still_fails(tables):
    reference, enrichment = tables
    changed = [{**enrichment[0], "Gene_set": "wrong_library"}]
    assert score(reference, changed, reference, enrichment).score == 0


@pytest.fixture
def scientific():
    if os.environ.get("DEG_GUEST_SCIENTIFIC") != "1":
        pytest.skip("Scientific inference tests run only in the isolated DEG diagnostic guest")
    import numpy
    from tasks.life_sciences.gene_expression_differential_analysis_functional_enrichment_analysis_1.runtime.deg_inference import (
        coefficients,
        inference,
    )

    return numpy, coefficients, inference


@pytest.mark.parametrize("beta", [[0.2, 0.3], [-2.0, 3.0], [-3.0, 0.2]])
def test_corrected_gradient_matches_objective_with_and_without_clipping(scientific, beta):
    numpy, probe, _ = scientific
    from scipy.optimize._numdiff import approx_derivative

    counts = numpy.array([0, 1, 4, 12])
    design = numpy.array([[1, 0], [1, 0], [1, 1], [1, 1]])
    factors = numpy.array([0.8, 1.2, 1.1, 0.9])
    beta = numpy.array(beta)
    arguments = (counts, design, factors, 0.3)
    actual = probe.coefficient_objective(beta, *arguments)[1]
    expected = approx_derivative(
        lambda value: probe.coefficient_objective(value, *arguments)[0], beta
    )
    numpy.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-8)


def test_centered_objective_preserves_pinned_nb_and_ridge_differences(scientific):
    numpy, probe, _ = scientific
    from pydeseq2 import utils

    counts = numpy.array([0, 1, 4, 12])
    design = numpy.array([[1, 0], [1, 0], [1, 1], [1, 1]])
    factors = numpy.ones(4)
    initial, final = numpy.array([-2.0, 3]), numpy.array([0.2, 0.3])

    def pinned(beta):
        means = numpy.maximum(factors * numpy.exp(design @ beta), 0.5)
        return utils.nb_nll(counts, means, 0.3) + 0.5e-6 * numpy.sum(beta**2)

    expected = pinned(final) - pinned(initial)
    actual = (
        probe.coefficient_objective(final, counts, design, factors, 0.3)[0]
        - probe.coefficient_objective(initial, counts, design, factors, 0.3)[0]
    )
    assert actual == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    "cr_reg,prior_reg", [(False, False), (False, True), (True, False), (True, True)]
)
def test_dispersion_optimization_preserves_regularization_switches(
    scientific, monkeypatch, cr_reg, prior_reg
):
    numpy, _, inference = scientific
    from types import SimpleNamespace

    counts = numpy.array([0, 1, 4, 12])
    design = numpy.array([[1, 0], [1, 0], [1, 1], [1, 1]])
    means = numpy.array([0.5, 1.0, 4.0, 8.0])

    def inspect_objective(objective, **kwargs):
        for dispersion in (0.01, 0.3, 2.0):
            expected = inference.stable_nb_nll(counts, means, dispersion)
            if cr_reg:
                weights = means / (1 + means * dispersion)
                expected += 0.5 * numpy.linalg.slogdet((design.T * weights) @ design)[1]
            if prior_reg:
                expected += (numpy.log(dispersion) - numpy.log(0.2)) ** 2 / (2 * 0.4)
            assert objective(numpy.log(dispersion)) == pytest.approx(expected, abs=1e-12)
        return SimpleNamespace(success=True, fun=objective(numpy.log(0.3)), x=numpy.log(0.3))

    monkeypatch.setattr(inference, "minimize_scalar", inspect_objective)
    inference.fit_dispersion(counts, design, means, 0.2, 1e-8, 10, 0.4, cr_reg, prior_reg)


@pytest.mark.parametrize("variance", [None, 0, -1, math.nan, math.inf])
def test_missing_or_invalid_map_prior_is_not_silently_dropped(scientific, variance):
    numpy, _, inference = scientific
    with pytest.raises(ValueError, match="prior variance"):
        inference.fit_dispersion(
            numpy.ones(4), numpy.ones((4, 1)), numpy.ones(4), 0.2, 1e-8, 10, variance, True, True
        )


@pytest.mark.parametrize("defect", ["failed", "nonstationary", "infeasible", "nonfinite"])
def test_uncertified_fallback_never_reports_convergence(scientific, monkeypatch, defect):
    numpy, _, inference = scientific
    fit = {"success": True, "objective": 1, "kkt_residual": 0, "minimum_slack": 0, "beta": [0, 0]}
    if defect == "failed":
        fit["success"] = False
    elif defect == "nonstationary":
        fit["kkt_residual"] = 0.1
    elif defect == "infeasible":
        fit["minimum_slack"] = -0.1
    else:
        fit["kkt_residual"] = math.nan
    monkeypatch.setattr(inference, "fit_regions", lambda *args: [fit])
    monkeypatch.setattr(inference, "fit_all_regions", lambda *args: [fit])
    design = numpy.array([[1, 0], [1, 1]] * 4)
    with pytest.raises(RuntimeError, match="not certified"):
        inference.fit_coefficients(
            numpy.ones(8), numpy.ones(8), design, 0.3, 0.5, 1e-8, -30, 30, "L-BFGS-B", 250
        )


@pytest.mark.parametrize("dispersion", [1e-8, 1e-5, 0.01, 0.1, 0.10001, 1.0, 10.0])
@pytest.mark.parametrize("large_counts", [False, True])
def test_stable_likelihood_matches_high_precision_integer_nb_identity(
    scientific, dispersion, large_counts
):
    numpy, _, inference = scientific
    from decimal import Decimal, localcontext

    counts = numpy.array([0, 1, 100, 1000] if large_counts else [0, 1, 4, 12])
    means = numpy.array([0.5, 1.0, 90.0, 1200.0] if large_counts else [0.5, 1.0, 4.0, 8.0])
    with localcontext() as context:
        context.prec = 60
        alpha = Decimal(str(dispersion))
        expected = Decimal(0)
        for count, mean in zip(counts, means):
            mean = Decimal(str(mean))
            expected += Decimal(math.factorial(int(count))).ln()
            expected -= sum((1 + alpha * value).ln() for value in range(int(count)))
            expected += (int(count) + 1 / alpha) * (1 + alpha * mean).ln()
            expected -= int(count) * mean.ln()
    assert inference.stable_nb_nll(counts, means, dispersion) == pytest.approx(
        float(expected), abs=1e-11, rel=0
    )


def test_sparse_fit_requires_a_lower_objective_and_valid_stationarity(scientific):
    numpy, probe, inference = scientific
    counts = numpy.array([0, 0, 0, 0, 0, 50, 1, 0])
    factors = numpy.array(
        [
            0.9958816241916957,
            0.9746539551504523,
            1.0823749011627233,
            0.9864693540476952,
            0.8217084391628742,
            0.8943965678149526,
            1.26729670160483,
            1.073946066062863,
        ]
    )
    design = numpy.array(
        [
            [1, 1, 0, 0, 0],
            [1, 0, 0, 0, 1],
            [1, 1, 0, 0, 1],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 1, 1],
            [1, 0, 0, 1, 0],
            [1, 0, 1, 0, 1],
            [1, 0, 1, 0, 0],
        ]
    )
    dispersion = 0.9687229450191825
    beta, means, hats, converged = inference.fit_coefficients(
        counts, factors, design, dispersion, 0.5, 1e-8, -30, 30, "L-BFGS-B", 250
    )
    assert converged and numpy.all(numpy.isfinite(hats))
    assert probe.coefficient_objective(beta, counts, design, factors, dispersion)[0] < 0.119250088
    numpy.testing.assert_allclose(means, factors * numpy.exp(design @ beta), rtol=1e-14)


def test_convex_zero_epigraph_matches_full_region_enumeration(scientific):
    numpy, probe, _ = scientific
    counts = numpy.array([1, 0, 1, 1, 0, 0, 30, 0])
    design = numpy.array([[1, 0], [1, 0], [1, 0], [1, 0], [1, 1], [1, 1], [1, 1], [1, 1]])
    factors = numpy.ones(8)
    initial = numpy.linalg.lstsq(design, numpy.log(counts + 0.1), rcond=None)[0]
    full = probe.fit_regions(counts, design, factors, 0.3, initial)[0]
    reduced = probe.fit_convex_regions(counts, design, factors, 0.3, initial)[0]
    assert reduced["success"] and reduced["kkt_residual"] < 1e-5
    assert reduced["objective"] == pytest.approx(full["objective"], abs=1e-9)
    numpy.testing.assert_allclose(reduced["beta"], full["beta"], atol=1e-5, rtol=0)
