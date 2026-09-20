"""Independent count-contract tests; diagnostic artifacts are never segmentation gold."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import asyncio

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from tasks.life_sciences.merfish_image_decoding_segmentation_1.scripts import (
    score_merfish as scorer,
)


GENES = ["Gene-A", "Gene-B", "Gene-C"]


def encode_tiff(mask):
    buffer = io.BytesIO()
    Image.fromarray(mask).save(buffer, format="TIFF")
    return buffer.getvalue()


def make_decoded(counts=(1000, 2000, 3000), blanks=100):
    names = np.repeat([*GENES, "Blank-1"], [*counts, blanks])
    positions = np.arange(len(names)) % 50
    return pd.DataFrame(
        {
            "gene": names,
            "x": 42 + positions * 4,
            "y": np.full(len(names), 42),
            "is_exact": np.ones(len(names), dtype=bool),
            "total_magnitude": np.full(len(names), 0.002),
        }
    )


@pytest.fixture
def case():
    mask = np.zeros(scorer.SEGMENTATION_SHAPE, dtype=np.uint16)
    for index in range(50):
        mask[40:44, 40 + index * 4 : 40 + (index + 1) * 4] = index + 1
    return {
        "decoded": make_decoded(),
        "reference": make_decoded(),
        "mask": mask,
        "codebook": {"mappings": [{"target": gene} for gene in [*GENES, "Blank-1"]]},
    }


def payloads(case):
    decoded = case["decoded"]
    mask = case["mask"]
    real = decoded[decoded.gene.isin(GENES)]
    labels = mask[np.round(real.y).astype(int), np.round(real.x).astype(int)]
    matrix = pd.crosstab(labels, real.gene.to_numpy()).reindex(
        index=np.arange(1, int(mask.max()) + 1), columns=GENES, fill_value=0
    )
    matrix.index.name = "cell_id"
    metrics = {
        "total_decoded_transcripts": len(decoded),
        "blank_rate": float(decoded.gene.str.startswith("Blank-").mean()),
        "exact_match_fraction": 1.0,
        "n_cells": int(mask.max()),
        "assigned_fraction": float((labels > 0).mean()),
        "mean_transcripts_per_cell": float(matrix.to_numpy().sum() / max(1, mask.max())),
    }
    return {
        "decoded_csv_bytes": decoded.to_csv(index=False).encode(),
        "segmentation_tiff_bytes": encode_tiff(mask),
        "cell_by_gene_csv_bytes": matrix.to_csv().encode(),
        "quality_metrics_json_bytes": json.dumps(metrics).encode(),
        "reference_csv_bytes": case["reference"].to_csv(index=False).encode(),
        "codebook_json_bytes": json.dumps(case["codebook"]).encode(),
    }


def test_independent_full_positive(case):
    result = scorer.score(**payloads(case))
    assert result.score == 1.0
    assert not result.hard_failed
    assert result.pearson_r == pytest.approx(1)
    assert result.spatial_concordance == 1
    assert result.structural["n_cells_segmentation"] == 50
    assert result.metrics_agent["total_decoded_transcripts"] == 6100
    assert "total_real=6000 reference_real=6000" in result.components["total_count"].detail


@pytest.mark.parametrize("encoding", ["row_order", "column_order", "scientific", "crlf", "quoted"])
def test_equivalent_decoded_serializations(case, encoding):
    frame = case["decoded"]
    arguments = payloads(case)
    if encoding == "row_order":
        encoded = frame.sample(frac=1, random_state=12).to_csv(index=False)
    elif encoding == "column_order":
        encoded = frame[list(reversed(frame.columns))].to_csv(index=False)
    elif encoding == "scientific":
        encoded = frame.astype({"x": float, "y": float}).to_csv(index=False, float_format="%.9e")
    elif encoding == "crlf":
        encoded = frame.to_csv(index=False, lineterminator="\r\n")
    else:
        encoded = frame.to_csv(index=False, quoting=1)
    arguments["decoded_csv_bytes"] = encoded.encode()
    assert scorer.score(**arguments).score == 1.0


@pytest.mark.parametrize("real_count,expected", [(4799, 0), (4800, 0.1), (7200, 0.1), (7201, 0)])
@pytest.mark.parametrize("blanks", [0, 100, 400])
def test_real_count_boundary_is_independent_of_blank_rows(case, real_count, expected, blanks):
    counts = [real_count // 6, real_count // 3]
    counts.append(real_count - sum(counts))
    case["decoded"] = make_decoded(counts, blanks)
    result = scorer.score(**payloads(case))
    assert result.pearson_r > 0.9
    assert result.components["total_count"].score == expected
    assert result.metrics_agent["total_decoded_transcripts"] == real_count + blanks


@pytest.mark.parametrize("padding_gene", ["Blank-1", "NotInCodebook"])
def test_padding_cannot_make_insufficient_real_transcripts_plausible(case, padding_gene):
    case["decoded"] = make_decoded((780, 1560, 2340), 1400)
    case["decoded"].loc[case["decoded"].gene == "Blank-1", "gene"] = padding_gene
    result = scorer.score(**payloads(case))
    assert result.pearson_r > 0.9
    assert result.components["total_count"].score == 0


def test_reference_blanks_do_not_change_real_count_target(case):
    baseline = scorer.score(**payloads(case))
    case["reference"] = make_decoded(blanks=12000)
    result = scorer.score(**payloads(case))
    assert result.components["total_count"] == baseline.components["total_count"]
    assert result.score == baseline.score


def test_reference_count_is_derived_not_a_fixture_constant(case):
    case["reference"] = make_decoded((1500, 3000, 4500))
    case["decoded"] = make_decoded((1500, 3000, 4500))
    result = scorer.score(**payloads(case))
    assert result.score == 1
    assert "reference_real=9000" in result.components["total_count"].detail


def test_invalid_real_gene_reference_is_not_replaced_with_a_magic_count(case):
    case["reference"] = make_decoded((0, 0, 0), 10)
    with pytest.raises(ValueError, match="no real-gene transcripts"):
        scorer.score(**payloads(case))


def test_wrong_gene_distribution_still_fails_all_decoding_credit(case):
    case["decoded"] = make_decoded((3000, 2000, 1000))
    result = scorer.score(**payloads(case))
    assert result.pearson_r < 0.5
    assert all(result.components[name].score == 0 for name in scorer.DECODING_CREDIT_COMPONENTS)


def test_wrong_locations_do_not_receive_spatial_credit(case):
    case["decoded"]["y"] = 1000
    result = scorer.score(**payloads(case))
    assert result.components["pearson"].score == 0.3
    assert result.components["total_count"].score == 0.1
    assert result.components["spatial"].score == 0
    assert result.structural["assigned_fraction"] == 0


def test_out_of_bounds_coordinates_do_not_get_clipped_into_a_cell(case):
    case["decoded"].loc[0, ["x", "y"]] = [-1, -1]
    result = scorer.score(**payloads(case))
    assert result.structural["invalid_coordinate_rows"] == 1
    assert result.structural["assigned_fraction"] < 1


def test_spatial_concordance_does_not_reuse_one_reference_transcript():
    agent = pd.DataFrame({"gene": ["Gene-A", "Gene-A"], "x": [10.0, 10.0], "y": [10.0, 10.0]})
    reference = pd.DataFrame({"gene": ["Gene-A"], "x": [10.0], "y": [10.0]})
    assert scorer._spatial_concordance(agent, reference, max_px=3.0) == 0.5


def test_empty_segmentation_and_wrong_genes_score_zero(case):
    case["mask"][:] = 0
    case["decoded"] = make_decoded((3000, 2000, 1000))
    result = scorer.score(**payloads(case))
    assert result.score == 0


def test_no_unjustified_segmentation_threshold_changes():
    assert (scorer.NUCLEUS_COUNT_MIN, scorer.NUCLEUS_COUNT_MAX) == (10, 60)
    assert (scorer.CELL_COUNT_MIN, scorer.CELL_COUNT_MAX) == (10, 60)
    assert scorer.ASSIGNED_MIN == 0.1
    assert (scorer.MEAN_PER_CELL_MIN, scorer.MEAN_PER_CELL_MAX) == (50, 1000)
    assert scorer.TOTAL_COUNT_TOL == 0.2
    assert (scorer.SPATIAL_MATCH_PX, scorer.SPATIAL_FULL, scorer.SPATIAL_PARTIAL) == (3, 0.8, 0.6)
    assert (scorer.BLANK_RATE_FULL, scorer.BLANK_RATE_PARTIAL) == (0.05, 0.07)


@pytest.mark.parametrize("nuclei", [17, 28])
def test_dapi_nucleus_counts_are_accepted_without_whole_cell_expansion(case, nuclei):
    case["mask"][case["mask"] > nuclei] = 0
    result = scorer.score(**payloads(case))
    assert result.components["cell_count"].score == scorer.WEIGHT_CELLCOUNT
    assert result.components["assigned_fraction"].score == scorer.WEIGHT_ASSIGNED
    assert result.components["mean_per_cell"].score == scorer.WEIGHT_MEAN
    assert result.components["matrix_consistency"].score == scorer.WEIGHT_CONSISTENCY


def test_too_few_nuclei_do_not_receive_segmentation_plausibility_credit(case):
    case["mask"][case["mask"] > 9] = 0
    result = scorer.score(**payloads(case))
    assert result.components["cell_count"].score == 0
    assert result.components["assigned_fraction"].score == 0
    assert result.components["mean_per_cell"].score == 0


@pytest.mark.parametrize("column", scorer.REQUIRED_DECODED_COLUMNS)
def test_missing_decoded_columns_still_fail(case, column):
    arguments = payloads(case)
    arguments["decoded_csv_bytes"] = (
        case["decoded"].drop(columns=column).to_csv(index=False).encode()
    )
    assert scorer.score(**arguments).hard_failed


@pytest.mark.parametrize("key", scorer.REQUIRED_METRIC_KEYS)
def test_missing_metrics_still_fail(case, key):
    arguments = payloads(case)
    metrics = json.loads(arguments["quality_metrics_json_bytes"])
    del metrics[key]
    arguments["quality_metrics_json_bytes"] = json.dumps(metrics).encode()
    assert scorer.score(**arguments).hard_failed


@pytest.mark.parametrize(
    "field,bad_bytes",
    [
        ("decoded_csv_bytes", b"gene,x,y,is_exact,total_magnitude\n"),
        ("decoded_csv_bytes", b'gene,x,y,is_exact,total_magnitude\n"unterminated'),
        ("segmentation_tiff_bytes", b"not an image"),
        ("cell_by_gene_csv_bytes", b"wrong,columns\n1,2\n"),
        ("quality_metrics_json_bytes", b'{"n_cells":'),
    ],
)
def test_malformed_outputs_still_fail(case, field, bad_bytes):
    arguments = payloads(case)
    arguments[field] = bad_bytes
    assert scorer.score(**arguments).hard_failed


@pytest.mark.parametrize("shape,dtype", [((10, 10), "uint16"), ((2048, 2048), "float32")])
def test_invalid_mask_shape_or_dtype_still_fails(case, shape, dtype):
    arguments = payloads(case)
    arguments["segmentation_tiff_bytes"] = encode_tiff(np.zeros(shape, dtype=dtype))
    assert scorer.score(**arguments).hard_failed


@pytest.mark.parametrize("coordinate", ["nan", "inf", "-inf", "not-a-number"])
def test_malformed_coordinates_receive_zero_at_evaluator_boundary(case, coordinate):
    from tasks.life_sciences.merfish_image_decoding_segmentation_1 import main

    arguments = payloads(case)
    decoded = case["decoded"].astype({"x": str})
    decoded.loc[0, "x"] = coordinate
    arguments["decoded_csv_bytes"] = decoded.to_csv(index=False).encode()
    metadata = {
        "decoded_file": "decoded_csv_bytes",
        "segmentation_file": "segmentation_tiff_bytes",
        "cell_by_gene_file": "cell_by_gene_csv_bytes",
        "metrics_file": "quality_metrics_json_bytes",
        "reference_file": "reference_csv_bytes",
        "input_codebook": "codebook_json_bytes",
    }

    class Session:
        async def read_bytes(self, path):
            return arguments[path]

    assert asyncio.run(main.evaluate(SimpleNamespace(metadata=metadata), Session())) == [0.0]


def test_public_contract_alignment():
    from tasks.life_sciences.merfish_image_decoding_segmentation_1 import main

    runtime = main.MerfishTaskConfig().task_description
    card = json.loads(Path(main.__file__).with_name("task_card.json").read_text())
    assert (
        runtime.split("5. Run Cellpose")[1].split("6. Assign")[0]
        == card["taskPrompt"].split("5. Run Cellpose")[1].split("6. Assign")[0]
    )
    assert (
        runtime.split("Total-count plausibility")[1]
        == card["taskPrompt"].split("Total-count plausibility")[1]
    )
    assert "diameter=None, channels=[0, 0]" in runtime
    assert "historical `cell` names" in runtime
    assert "nuclear transcript counts" in runtime
    assert "reference_real" in card["evaluation"]
    assert "per-cell/per-gene counts" in card["evaluation"]
    assert "each positive segmentation label exactly once" in card["evaluation"]
    assert "nonnegative integer counts" in card["evaluation"]
    assert "column sums tie out" not in card["evaluation"]
    assert "around 30-50" not in " ".join(card["agentMustDo"])


def test_recovered_historical_artifact_static_rescore():
    diagnostic_dir = os.environ.get("MERFISH_AUDIT_DIR")
    if not diagnostic_dir:
        pytest.skip("set MERFISH_AUDIT_DIR to opt into recovered-artifact static scoring")
    root = Path(diagnostic_dir)
    before = json.loads((root / "historical-score-before.json").read_text())
    base = Path(__file__).resolve().parents[2] / (
        "task-data-hf/extracted/life_sciences/merfish_image_decoding_segmentation_1/base"
    )
    import hashlib

    for filename, digest in before["artifact_hashes"].items():
        assert hashlib.sha256((root / "historical" / filename).read_bytes()).hexdigest() == digest
    result = scorer.score(
        decoded_csv_bytes=(root / "historical/decoded_transcripts.csv").read_bytes(),
        segmentation_tiff_bytes=(root / "historical/segmentation.tiff").read_bytes(),
        cell_by_gene_csv_bytes=(root / "historical/cell_by_gene.csv").read_bytes(),
        quality_metrics_json_bytes=(root / "historical/quality_metrics.json").read_bytes(),
        reference_csv_bytes=(base / "reference/benchmark_results.csv").read_bytes(),
        codebook_json_bytes=(base / "input/codebook.json").read_bytes(),
    )
    assert before["score"] == 0.425
    assert result.score == 0.75
    assert result.structural["n_cells_segmentation"] == 28
    assert result.components["cell_count"].score == scorer.WEIGHT_CELLCOUNT
    assert result.components["assigned_fraction"].score == scorer.WEIGHT_ASSIGNED
    assert result.components["mean_per_cell"].score == scorer.WEIGHT_MEAN
    assert result.components["total_count"].score == 0
    assert "total_real=6770 reference_real=42212" in result.components["total_count"].detail
    assert result.structural["decoded_real_outside_reference_support"] == 512


@pytest.fixture
def matrix_case(case):
    case["decoded"] = make_decoded((1001, 2002, 3003))
    return payloads(case)


@pytest.mark.parametrize("encoding", ["rows", "labels", "scientific", "decimal"])
def test_equivalent_matrices_retain_credit(matrix_case, encoding):
    baseline = scorer.score(**matrix_case)
    matrix = pd.read_csv(io.BytesIO(matrix_case["cell_by_gene_csv_bytes"]))
    if encoding == "labels":
        mask = np.array(Image.open(io.BytesIO(matrix_case["segmentation_tiff_bytes"])))
        mask[mask > 0] = 65535 - mask[mask > 0] * 7
        matrix["cell_id"] = 65535 - matrix["cell_id"] * 7
        matrix_case["segmentation_tiff_bytes"] = encode_tiff(mask)
    if encoding in {"rows", "labels"}:
        matrix = matrix.sample(frac=1, random_state=17)
        encoded = matrix.to_csv(index=False)
    else:
        encoded = matrix.astype(float).to_csv(
            index=False, float_format="%.12e" if encoding == "scientific" else "%.1f"
        )
    matrix_case["cell_by_gene_csv_bytes"] = encoded.encode()
    result = scorer.score(**matrix_case)
    assert not result.hard_failed
    assert result.score == baseline.score == 1.0
    assert result.components == baseline.components


@pytest.mark.parametrize(
    "mutation", ["allocations", "ids", "duplicate_ids", "fractional", "negative"]
)
def test_wrong_matrix_loses_only_consistency_credit(matrix_case, mutation):
    baseline = scorer.score(**matrix_case)
    matrix = pd.read_csv(io.BytesIO(matrix_case["cell_by_gene_csv_bytes"]))
    original_totals = matrix[GENES].sum()
    if mutation == "allocations":
        matrix[GENES] = matrix[GENES].iloc[::-1].to_numpy()
    elif mutation == "ids":
        matrix["cell_id"] = matrix["cell_id"].iloc[::-1].to_numpy()
    elif mutation == "duplicate_ids":
        matrix.loc[0, "cell_id"] = matrix.loc[1, "cell_id"]
    elif mutation == "fractional":
        matrix = matrix.astype({GENES[0]: float})
        matrix.loc[0, GENES[0]] += 0.25
    else:
        transferred = matrix.loc[0, GENES[0]] + 1
        matrix.loc[0, GENES[0]] -= transferred
        matrix.loc[1, GENES[0]] += transferred
    if mutation != "fractional":
        pd.testing.assert_series_equal(matrix[GENES].sum(), original_totals)
    matrix_case["cell_by_gene_csv_bytes"] = matrix.to_csv(index=False).encode()
    result = scorer.score(**matrix_case)
    assert not result.hard_failed
    assert result.score == 0.9
    assert result.components["matrix_consistency"].score == 0
    for name in baseline.components.keys() - {"matrix_consistency"}:
        assert result.components[name].score == baseline.components[name].score


@pytest.mark.parametrize("column", ["cell_id", "Gene-A"])
@pytest.mark.parametrize(
    "value",
    ["", "NaN", "inf", "-inf", "text", "True", "-1", "1.25", "1e10000", "9223372036854775808"],
)
def test_invalid_matrix_numbers_do_not_erase_independent_credit(matrix_case, column, value):
    baseline = scorer.score(**matrix_case)
    matrix = pd.read_csv(io.BytesIO(matrix_case["cell_by_gene_csv_bytes"]), dtype=str)
    matrix.loc[0, column] = value
    matrix_case["cell_by_gene_csv_bytes"] = matrix.to_csv(index=False).encode()
    result = scorer.score(**matrix_case)
    assert not result.hard_failed
    assert result.components["matrix_consistency"].score == 0
    for name in [*scorer.DECODING_CREDIT_COMPONENTS, "cell_count", "assigned_fraction"]:
        assert result.components[name] == baseline.components[name]


@pytest.mark.parametrize("column", ["cell_id", "Gene-A"])
def test_near_integer_text_is_not_rounded_into_valid_counts_or_ids(matrix_case, column):
    matrix = pd.read_csv(io.BytesIO(matrix_case["cell_by_gene_csv_bytes"]), dtype=str)
    matrix.loc[0, column] += ".000000000000000000001"
    matrix_case["cell_by_gene_csv_bytes"] = matrix.to_csv(index=False).encode()
    result = scorer.score(**matrix_case)
    assert not result.hard_failed
    assert result.components["matrix_consistency"].score == 0


@pytest.mark.parametrize("mutation", ["zero_id", "unknown_id", "missing_row", "extra_row"])
def test_matrix_ids_must_cover_exactly_the_positive_mask_labels(matrix_case, mutation):
    matrix = pd.read_csv(io.BytesIO(matrix_case["cell_by_gene_csv_bytes"]))
    if mutation in {"zero_id", "unknown_id"}:
        matrix.loc[0, "cell_id"] = 0 if mutation == "zero_id" else 999
    elif mutation == "missing_row":
        matrix = matrix.iloc[1:]
    else:
        matrix = pd.concat([matrix, matrix.iloc[:1]], ignore_index=True)
    matrix_case["cell_by_gene_csv_bytes"] = matrix.to_csv(index=False).encode()
    result = scorer.score(**matrix_case)
    assert not result.hard_failed
    assert result.components["matrix_consistency"].score == 0


def test_matrix_gene_column_order_remains_required(matrix_case):
    matrix = pd.read_csv(io.BytesIO(matrix_case["cell_by_gene_csv_bytes"]))
    matrix_case["cell_by_gene_csv_bytes"] = (
        matrix[["cell_id", *reversed(GENES)]].to_csv(index=False).encode()
    )
    assert scorer.score(**matrix_case).hard_failed


def test_zero_transcript_cells_are_required_and_background_blanks_excluded(case):
    case["mask"][40:44, 240:244] = 51
    case["decoded"].loc[0, "y"] = 10
    arguments = payloads(case)
    result = scorer.score(**arguments)
    matrix = pd.read_csv(io.BytesIO(arguments["cell_by_gene_csv_bytes"]))
    assert matrix.loc[matrix.cell_id == 51, GENES].to_numpy().sum() == 0
    assert matrix[GENES].to_numpy().sum() == 5999
    assert result.components["matrix_consistency"].score == 0.1


def test_matrix_uses_existing_rounding_and_xy_order(case):
    case["decoded"] = case["decoded"].astype({"x": float})
    case["decoded"].loc[:3, "x"] = [43.49, 43.5, 47.5, 239.49]
    result = scorer.score(**payloads(case))
    assert result.components["matrix_consistency"].score == 0.1


@pytest.mark.parametrize("axis", ["x", "y"])
def test_reference_region_has_explicit_inclusive_boundaries(axis):
    frame = pd.DataFrame({"x": [100.0] * 6, "y": [100.0] * 6})
    frame[axis] = [0.0, 39.999, 40.0, 2008.0, 2008.001, 2047.0]
    assert scorer._reference_support_mask(frame).tolist() == [
        False,
        False,
        True,
        True,
        False,
        False,
    ]


def test_unlabeled_border_is_excluded_only_from_reference_comparisons(case):
    baseline = scorer.score(**payloads(case))
    border = make_decoded((2500, 0, 0), 0)
    border["x"] = 10
    case["mask"][40:44, 8:12] = 1
    case["decoded"] = pd.concat([case["decoded"], border], ignore_index=True)
    arguments = payloads(case)
    result = scorer.score(**arguments)
    assert result.score == baseline.score == 1.0
    for name in ("pearson", "total_count", "spatial"):
        assert result.components[name] == baseline.components[name]
    assert result.structural["decoded_real_in_reference_support"] == 6000
    assert result.structural["decoded_real_outside_reference_support"] == 2500
    assert result.metrics_agent["total_decoded_transcripts"] == 8600
    assert result.structural["mean_transcripts_per_cell"] == 170
    matrix = pd.read_csv(io.BytesIO(arguments["cell_by_gene_csv_bytes"]))
    assert matrix[GENES].to_numpy().sum() == 8500
    assert result.components["matrix_consistency"].score == 0.1


def test_reference_and_submission_use_the_same_region(case):
    border = make_decoded((0, 0, 9000), 0)
    border["y"] = 2010
    case["reference"] = pd.concat([case["reference"], border], ignore_index=True)
    assert scorer.score(**payloads(case)).score == 1.0


def test_border_only_calls_cannot_receive_decoding_credit(case):
    case["decoded"]["y"] = 20
    result = scorer.score(**payloads(case))
    assert result.pearson_r == 0
    assert result.spatial_concordance == 0
    assert all(result.components[name].score == 0 for name in scorer.DECODING_CREDIT_COMPONENTS)


def test_border_calls_do_not_rescue_incorrect_interior_gene_distribution(case):
    case["decoded"] = make_decoded((3000, 2000, 1000))
    border = make_decoded((0, 0, 10000), 0)
    border["y"] = 20
    case["decoded"] = pd.concat([case["decoded"], border], ignore_index=True)
    result = scorer.score(**payloads(case))
    assert result.pearson_r < 0.5
    assert all(result.components[name].score == 0 for name in scorer.DECODING_CREDIT_COMPONENTS)


def test_public_workflow_preserves_the_binary_codebook():
    from tasks.life_sciences.merfish_image_decoding_segmentation_1 import main

    description = main.MerfishTaskConfig().task_description
    card = json.loads(Path(main.__file__).with_name("task_card.json").read_text())
    for prompt in (description, card["taskPrompt"]):
        assert "full 16-component intensity traces" in prompt
        assert "DetectPixels.PixelSpotDecoder" in prompt
        assert "full 16-bit Hamming decoder accepting at most one bit error" in prompt
        assert "Do not use native 8-round `CheckAll`" in prompt
        assert "dividing each tile" in prompt
        assert "40 <= x <= 2008" in prompt and "40 <= y <= 2008" in prompt
        assert "do not crop the mask or translate coordinates" in prompt
    assert "MHD4-aware `CheckAll`" not in " ".join(card["agentMustDo"])
