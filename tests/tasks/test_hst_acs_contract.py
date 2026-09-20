"""Independent HST report-equivalence and public QC contract regressions."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from tasks.physical_sciences.hst_acs_wfc_visit_reduction.scripts import score_outputs as scorer


QC = {
    "visit_id": "synthetic_visit",
    "filter": "TEST",
    "num_sources": 3,
    "background_median_e_s": 0.004,
    "cosmic_ray_pixels_masked": 11,
    "hot_pixels_masked": 7,
    "aperture_radius_pix": 3.0,
    "pixfrac": 0.8,
    "final_scale_arcsec_per_pix": 0.05,
    "astrometric_rms_pix": 0.02,
}
REPORT = (
    "Used calacs-style calibration and AstroDrizzle-style coaddition. "
    "Astrometric RMS was measured against anchors; cosmic ray masking used the DQ bits."
)


def write_table(path, fields, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def visits(tmp_path):
    reference = tmp_path / "reference" / "synthetic_visit"
    output = tmp_path / "output" / "synthetic_visit"
    sources = [
        {
            "source_id": f"source{index}",
            "x": 10 + index * 8,
            "y": 12 + index * 9,
            "ra_deg": 130 + index / 100,
            "dec_deg": 4 + index / 100,
            "flux_e_s": 100 + index * 10,
            "mag_ab": 21 + index / 10,
            "snr": 50,
            "sharpness": 0.2,
            "flags": 0,
        }
        for index in range(3)
    ]
    alignment = [
        {
            "exposure_id": f"exposure{index}",
            "dx_pix": index / 4,
            "dy_pix": -index / 4,
            "rms_pix": 0.02,
            "matched_sources": 3,
        }
        for index in range(2)
    ]
    for root in (reference, output):
        root.mkdir(parents=True)
        write_table(root / "source_catalog.csv", scorer.SOURCE_FIELDS, sources)
        write_table(root / "alignment_solution.csv", scorer.ALIGN_FIELDS, alignment)
        (root / "photometry_qc.json").write_text(json.dumps(QC))
        (root / "reduction_report.md").write_text(REPORT)
        np.savetxt(root / "drizzled_image.csv", np.zeros((72, 72)), delimiter=",")
    return output, reference


@pytest.mark.parametrize(
    "separator",
    [
        " ",
        "-",
        " - ",
        "\n\t",
        "\u2010",
        "\u2011",
        "\u2012",
        "\u2013",
        "\u2014",
        "\u2015",
        "\u2212",
        "\u00ad",
        "\u00a0",
        "\ufe63",
        "\uff0d",
        "\u058a",
    ],
)
@pytest.mark.parametrize("case", ["lower", "upper", "title"])
def test_all_report_phrases_accept_equivalent_separators_and_case(separator, case):
    terms = ["calacs style", "astrodrizzle style", "astrometric rms", "cosmic ray"]
    report = "; ".join(getattr(term.replace(" ", separator), case)() for term in terms)
    assert scorer._report_phrase_score(report) == 6.0


@pytest.mark.parametrize(
    "report",
    [
        "",
        "A beautiful image with accurate photometry.",
        "calibration; drizzle; astrometry; masking",
        "calacs; style; astrodrizzle; astrometric; rms",
        "uncalacs-style nonastrodrizzle-style nonastrometric rms cosmic raytracing",
        "calacs-stylesheet astrodrizzle-stylesheet astrometric rmsfake cosmic rayleigh",
        "calacs/style astrodrizzle/style astrometric/rms cosmic/ray",
        "calacs_style astrodrizzle_style astrometric_rms cosmic_ray",
    ],
)
def test_unrelated_or_partial_words_do_not_receive_report_credit(report):
    assert scorer._report_phrase_score(report) == 0.0


def test_plural_cosmic_rays_and_repeated_terms_do_not_change_weights():
    assert scorer._report_phrase_score("cosmic rays") == 1.5
    assert scorer._report_phrase_score(REPORT * 20) == 6.0
    assert scorer._report_phrase_score("calacs-style; ASTROMETRIC-RMS") == 3.0


def test_equivalent_report_scores_are_observable_below_total_score_cap(visits):
    output, reference = visits
    np.savetxt(output / "drizzled_image.csv", np.full((72, 72), 40), delimiter=",")
    before = scorer.score_visit(output, reference)[0]
    assert before == 94.0
    equivalent = "CALACS STYLE; astrodrizzle\nstyle; Astrometric-RMS; Cosmic\u2011ray masking."
    (output / "reduction_report.md").write_text(equivalent)
    assert scorer.score_visit(output, reference)[0] == before
    (output / "reduction_report.md").write_text("A polished but unrelated report.")
    assert scorer.score_visit(output, reference)[0] == before - 6.0


def test_independent_complete_fixture_and_directory_evaluation(visits):
    output, reference = visits
    assert scorer.score_visit(output, reference) == (100.0, [])
    result = scorer.evaluate_output_directory(output.parent, reference.parent)
    assert result["score"] == 1.0 and result["passed"]
    assert result["visit_scores"] == [100.0]
    assert scorer.PASS_THRESHOLD == 0.8


def test_subpixel_centroids_within_pixel_quantization_keep_full_credit(visits):
    output, reference = visits
    rows = scorer._csv_rows(output / "source_catalog.csv")
    for row in rows:
        row["x"] = str(float(row["x"]) + 0.30)
        row["y"] = str(float(row["y"]) - 0.20)
    write_table(output / "source_catalog.csv", scorer.SOURCE_FIELDS, rows)
    assert scorer.score_visit(output, reference) == (100.0, [])


def test_alignment_row_order_preserves_exposure_identity(visits):
    output, reference = visits
    np.savetxt(output / "drizzled_image.csv", np.full((72, 72), 40), delimiter=",")
    expected = scorer.score_visit(output, reference)[0]
    alignment = scorer._csv_rows(output / "alignment_solution.csv")
    write_table(output / "alignment_solution.csv", scorer.ALIGN_FIELDS, alignment[::-1])
    assert scorer.score_visit(output, reference)[0] == expected


@pytest.mark.parametrize(
    "filename", ["source_catalog.csv", "alignment_solution.csv", "drizzled_image.csv"]
)
def test_standard_csv_bom_preserves_score(visits, filename):
    output, reference = visits
    np.savetxt(output / "drizzled_image.csv", np.full((72, 72), 40), delimiter=",")
    expected = scorer.score_visit(output, reference)[0]
    path = output / filename
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    assert scorer.score_visit(output, reference)[0] == expected


@pytest.mark.parametrize(
    "filename", ["source_catalog.csv", "alignment_solution.csv", "drizzled_image.csv"]
)
def test_standard_quoted_csv_preserves_score(visits, filename):
    output, reference = visits
    np.savetxt(output / "drizzled_image.csv", np.full((72, 72), 40), delimiter=",")
    expected = scorer.score_visit(output, reference)[0]
    path = output / filename
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        csv.writer(handle, quoting=csv.QUOTE_ALL, lineterminator="\r\n").writerows(rows)
    assert scorer.score_visit(output, reference)[0] == expected


@pytest.mark.parametrize("change", ["duplicate", "unknown", "empty", "swapped"])
def test_alignment_ids_must_identify_the_correct_exposures(visits, change):
    output, reference = visits
    np.savetxt(output / "drizzled_image.csv", np.full((72, 72), 40), delimiter=",")
    alignment = scorer._csv_rows(output / "alignment_solution.csv")
    if change == "duplicate":
        alignment[1]["exposure_id"] = alignment[0]["exposure_id"]
    elif change == "unknown":
        alignment[1]["exposure_id"] = "unrelated_exposure"
    elif change == "empty":
        alignment[1]["exposure_id"] = ""
    else:
        alignment[0]["exposure_id"], alignment[1]["exposure_id"] = (
            alignment[1]["exposure_id"],
            alignment[0]["exposure_id"],
        )
    write_table(output / "alignment_solution.csv", scorer.ALIGN_FIELDS, alignment)
    score, notes = scorer.score_visit(output, reference)
    assert score == 84.0
    assert "alignment solution outside tolerance" in notes


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "not-a-number", ""])
def test_invalid_alignment_measurements_lose_alignment_credit_without_crashing(visits, value):
    output, reference = visits
    np.savetxt(output / "drizzled_image.csv", np.full((72, 72), 40), delimiter=",")
    alignment = scorer._csv_rows(output / "alignment_solution.csv")
    alignment[1]["dx_pix"] = value
    write_table(output / "alignment_solution.csv", scorer.ALIGN_FIELDS, alignment)
    score, notes = scorer.score_visit(output, reference)
    assert score == 84.0
    assert "alignment solution outside tolerance" in notes


@pytest.mark.parametrize(
    "key,value,loss",
    [
        ("cosmic_ray_pixels_masked", 12, 3),
        ("hot_pixels_masked", 6, 3),
        ("num_sources", 5, 3),
        ("num_sources", 4, 0),
        ("astrometric_rms_pix", 0.061, 3),
        ("astrometric_rms_pix", 0.059, 0),
        ("aperture_radius_pix", 2.0, 4),
        ("pixfrac", 1.0, 4),
        ("aperture_radius_pix", 3, 0),
        ("pixfrac", 0.80, 0),
    ],
)
def test_existing_qc_values_and_tolerances_are_preserved(visits, key, value, loss):
    output, reference = visits
    np.savetxt(output / "drizzled_image.csv", np.full((72, 72), 40), delimiter=",")
    (output / "photometry_qc.json").write_text(json.dumps({**QC, key: value}))
    assert scorer.score_visit(output, reference)[0] == 94.0 - loss


def test_wrong_or_nonfinite_qc_is_not_fixed_by_report_normalization(visits):
    output, reference = visits
    np.savetxt(output / "drizzled_image.csv", np.full((72, 72), 40), delimiter=",")
    qc = {key: (value if isinstance(value, str) else float("nan")) for key, value in QC.items()}
    (output / "photometry_qc.json").write_text(json.dumps(qc))
    assert scorer.score_visit(output, reference)[0] == 78.0
    assert not scorer.evaluate_output_directory(output.parent, reference.parent)["passed"]


def test_missing_canonical_qc_keys_still_lose_only_their_existing_points(visits):
    output, reference = visits
    np.savetxt(output / "drizzled_image.csv", np.full((72, 72), 40), delimiter=",")
    qc = dict(QC)
    qc["masked_cosmic_ray_pixels"] = qc.pop("cosmic_ray_pixels_masked")
    qc["masked_hot_pixels"] = qc.pop("hot_pixels_masked")
    (output / "photometry_qc.json").write_text(json.dumps(qc))
    assert scorer.score_visit(output, reference)[0] == 88.0


def test_report_phrases_cannot_make_wrong_science_pass(visits):
    output, reference = visits
    rows = scorer._csv_rows(output / "source_catalog.csv")
    for row in rows:
        row["x"] = "1000"
        row["flux_e_s"] = "1"
    write_table(output / "source_catalog.csv", scorer.SOURCE_FIELDS, rows)
    np.savetxt(output / "drizzled_image.csv", np.full((72, 72), 1000), delimiter=",")
    result = scorer.evaluate_output_directory(output.parent, reference.parent)
    assert result["score"] < 0.8
    assert not result["passed"]
    assert any("low source completeness" in note for note in result["notes"])
    assert any("RMSE too high" in note for note in result["notes"])


@pytest.mark.parametrize("filename", sorted(scorer.REQUIRED_VISIT_FILES))
def test_missing_outputs_still_fail(visits, filename):
    output, reference = visits
    (output / filename).unlink()
    assert scorer.score_visit(output, reference)[0] == 0.0


def test_malformed_json_still_fails(visits):
    output, reference = visits
    (output / "photometry_qc.json").write_text('{"num_sources":')
    score, notes = scorer.score_visit(output, reference)
    assert score == 0.0
    assert any("could not parse outputs" in note for note in notes)


def test_runtime_and_card_publish_the_same_qc_contract():
    from tasks.physical_sciences.hst_acs_wfc_visit_reduction import main

    config = main.HstAcsWfcVisitReductionConfig()
    description = config.task_description
    card = json.loads(Path(main.__file__).with_name("task_card.json").read_text())
    marker = "## Photometry QC Contract\n"
    contract = description.split(marker)[1].split("\nThe starter implementation")[0]
    assert contract == card["taskPrompt"].split(marker)[1].split("\nThe starter implementation")[0]
    for key in QC:
        assert f"| `{key}` |" in contract
    for required in (
        "3.0 output pixels",
        "pixfrac 0.8",
        "pixel_scale_arcsec",
        "finite",
        "DQ bit 4096",
        "DQ bit 16",
        "summed over exposures",
        "both counts",
        "not just recorded",
        "calacs-style",
        "AstroDrizzle-style",
        "astrometric RMS",
        "cosmic-ray masking",
        "Case, hyphens, and whitespace",
    ):
        assert required in contract
    assert config.prompt_file in description and config.visible_visit_dir in description
    assert "--input <visit_root_or_parent_input_dir> --output <output_dir>" in description
    assert config.candidate_script in description


@pytest.mark.parametrize(
    "filename",
    ["source_catalog.csv", "alignment_solution.csv", "photometry_qc.json", "drizzled_image.csv"],
)
def test_missing_reference_is_an_evaluator_error(visits, filename):
    output, reference = visits
    (reference / filename).unlink()
    with pytest.raises(RuntimeError, match="reference"):
        scorer.score_visit(output, reference)


@pytest.mark.parametrize(
    "filename,payload",
    [
        ("source_catalog.csv", "source_id,x\nbroken,nan\n"),
        ("alignment_solution.csv", "exposure_id,dx_pix\nbroken,nan\n"),
        ("photometry_qc.json", "{}"),
        ("photometry_qc.json", "[]"),
        ("drizzled_image.csv", "nan,nan\nnan,nan\n"),
    ],
)
def test_malformed_reference_is_an_evaluator_error(visits, filename, payload):
    output, reference = visits
    (reference / filename).write_text(payload)
    with pytest.raises(RuntimeError, match="reference"):
        scorer.score_visit(output, reference)


def test_empty_reference_directory_is_an_evaluator_error(tmp_path):
    reference = tmp_path / "reference"
    reference.mkdir()
    with pytest.raises(RuntimeError, match="reference"):
        scorer.evaluate_output_directory(tmp_path / "output", reference)


@pytest.mark.parametrize("value", ["invalid", None, [], {}])
def test_malformed_candidate_qc_remains_candidate_failure(visits, value):
    output, reference = visits
    (output / "photometry_qc.json").write_text(json.dumps({**QC, "num_sources": value}))
    assert scorer.score_visit(output, reference)[0] == 0.0


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("error,loss", [(0.0, 0), (0.079, 0), (0.081, 5), (0.179, 5), (0.181, 10)])
def test_visit_wide_shift_sign_preserves_tolerances(visits, sign, error, loss):
    output, reference = visits
    (output / "reduction_report.md").write_text("")
    alignment = scorer._csv_rows(output / "alignment_solution.csv")
    for row in alignment:
        row["dx_pix"] = sign * (float(row["dx_pix"]) + error)
        row["dy_pix"] = sign * float(row["dy_pix"])
    write_table(output / "alignment_solution.csv", scorer.ALIGN_FIELDS, alignment[::-1])
    score, notes = scorer.score_visit(output, reference)
    assert score == 100.0 - loss
    assert ("alignment solution outside tolerance" in notes) == (loss == 10)


@pytest.mark.parametrize(
    "change", ["axis_sign", "exposure_sign", "axis_swap", "magnitudes", "zero"]
)
def test_sign_equivalence_does_not_accept_incorrect_transforms(visits, change):
    output, reference = visits
    (output / "reduction_report.md").write_text("")
    alignment = scorer._csv_rows(output / "alignment_solution.csv")
    alignment[0].update(dx_pix=0.6, dy_pix=-0.8)
    alignment[1].update(dx_pix=-1.1, dy_pix=0.3)
    write_table(reference / "alignment_solution.csv", scorer.ALIGN_FIELDS, alignment)
    for index, row in enumerate(alignment):
        if change == "axis_sign":
            row["dx_pix"] *= -1
        elif change == "exposure_sign" and index == 1:
            row["dx_pix"] *= -1
            row["dy_pix"] *= -1
        elif change == "axis_swap":
            row["dx_pix"], row["dy_pix"] = row["dy_pix"], row["dx_pix"]
        elif change == "magnitudes":
            row["dx_pix"] += 0.3
        elif change == "zero":
            row["dx_pix"] = row["dy_pix"] = 0
    write_table(output / "alignment_solution.csv", scorer.ALIGN_FIELDS, alignment)
    score, notes = scorer.score_visit(output, reference)
    assert score == 90.0
    assert "alignment solution outside tolerance" in notes


def test_runtime_and_card_disclose_both_transform_conventions():
    from tasks.physical_sciences.hst_acs_wfc_visit_reduction import main

    description = main.HstAcsWfcVisitReductionConfig().task_description
    card = json.loads(Path(main.__file__).with_name("task_card.json").read_text())
    marker = "## Alignment Shift Convention\n"
    contract = description.split(marker)[1].split("## Photometry QC Contract")[0]
    assert contract == card["taskPrompt"].split(marker)[1].split("## Photometry QC Contract")[0]
    assert "x_mosaic = x_detector - dx_pix" in contract
    assert "x_mosaic = x_detector + dx_pix" in contract
    assert "one convention for both axes and every exposure" in contract
