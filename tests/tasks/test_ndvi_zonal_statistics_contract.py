import csv
from decimal import Decimal
from importlib import import_module

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

verifier = import_module("tasks.agriculture_env.ndvi_zonal_statistics_d02.scripts.verify_outputs")


@pytest.fixture
def bundles(tmp_path):
    for name in ("pred", "reference"):
        directory = tmp_path / name
        directory.mkdir()
        with (directory / verifier.CSV_FILENAME).open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(verifier.EXPECTED_CSV_COLUMNS)
            writer.writerows(
                [
                    ["001", "3", "3", "0.801170", "0.801170"],
                    ["002", "0", "0", "NA", "NA"],
                ]
            )
        with rasterio.open(
            directory / verifier.TIFF_FILENAME,
            "w",
            driver="GTiff",
            width=2,
            height=2,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=rasterio.transform.from_origin(10, 50, 0.01, 0.01),
        ) as dataset:
            dataset.write(np.array([[0.8, 0.80117], [0.9, np.nan]], dtype="float32"), 1)
    return tmp_path / "pred", tmp_path / "reference"


def replace_cell(directory, field, value, row=0):
    path = directory / verifier.CSV_FILENAME
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[row][field] = value
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=verifier.EXPECTED_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize(
    "token", ["0.801170", "0.80117", ".80117", "8.0117e-1", "+0.80117", "0.8011700"]
)
def test_numeric_equivalent_six_decimal_csv_has_no_penalty(bundles, token):
    pred, reference = bundles
    for field in verifier.CSV_ROUNDED_FIELDS:
        replace_cell(pred, field, token)
    report = verifier.evaluate_outputs(pred, reference)
    assert report["score"] == 1.0
    assert report["format_scores"]["format_penalty"] == 0.0


@pytest.mark.parametrize("token", ["0", "1", "-1", "-0", "-0.12345", "0.000001", "1e-6"])
def test_finite_rounded_boundaries_remain_valid(bundles, token):
    pred, reference = bundles
    for field in verifier.CSV_ROUNDED_FIELDS:
        replace_cell(pred, field, token)
        replace_cell(reference, field, f"{Decimal(token):.6f}")
    assert verifier.evaluate_outputs(pred, reference)["score"] == 1.0


@pytest.mark.parametrize("token", ["0.8011701", "0.8011699", "8.011701e-1"])
def test_nonzero_precision_beyond_six_decimals_still_penalized(bundles, token):
    pred, reference = bundles
    replace_cell(pred, "mean_ndvi", token)
    report = verifier.evaluate_outputs(pred, reference)
    assert report["csv_scores"]["mean_ndvi"] == 20.0
    assert report["format_scores"]["format_penalty"] == 10.0
    assert report["score"] == 0.9


@pytest.mark.parametrize("token", ["", "invalid", "NaN", "sNaN", "Infinity", "-Infinity", "1e999"])
def test_invalid_numeric_tokens_lose_accuracy_and_format_points(bundles, token):
    pred, reference = bundles
    replace_cell(pred, "mean_ndvi", token)
    report = verifier.evaluate_outputs(pred, reference)
    assert report["csv_scores"]["mean_ndvi"] == 10.0
    assert report["format_scores"]["format_penalty"] == 10.0
    assert report["score"] == 0.8


@pytest.mark.parametrize(
    ("field", "value", "row", "expected_points"),
    [
        ("mean_ndvi", "0.701170", 0, 10.0),
        ("median_ndvi", "NA", 0, 10.0),
        ("mean_ndvi", "0", 1, 10.0),
        ("rot_count", "4", 0, 5.0),
        ("valid_px", "4", 0, 5.0),
    ],
)
def test_scientific_value_and_zero_pixel_checks_remain(bundles, field, value, row, expected_points):
    pred, reference = bundles
    replace_cell(pred, field, value, row)
    report = verifier.evaluate_outputs(pred, reference)
    assert report["csv_scores"][field] == expected_points
    assert report["score"] < 1.0


@pytest.mark.parametrize(
    "damage", ["duplicate_id", "wrong_id", "crs", "transform", "dtype", "mask", "value"]
)
def test_schema_and_raster_checks_remain(bundles, damage):
    pred, reference = bundles
    if damage in {"duplicate_id", "wrong_id"}:
        replace_cell(pred, "id_lcp", "001" if damage == "duplicate_id" else "003", 1)
    else:
        path = pred / verifier.TIFF_FILENAME
        with rasterio.open(path) as dataset:
            array, profile = dataset.read(1), dataset.profile
        if damage == "crs":
            profile["crs"] = "EPSG:3857"
        elif damage == "transform":
            profile["transform"] = rasterio.transform.from_origin(11, 50, 0.01, 0.01)
        elif damage == "dtype":
            profile["dtype"] = "float64"
        elif damage == "mask":
            array[1, 1] = 0.8
        else:
            array[0, 0] = 0.5
        with rasterio.open(path, "w", **profile) as dataset:
            dataset.write(array, 1)
    report = verifier.evaluate_outputs(pred, reference)
    assert report["score"] < 1.0
    if damage not in {"mask", "value"}:
        assert not report["gate_passed"]
