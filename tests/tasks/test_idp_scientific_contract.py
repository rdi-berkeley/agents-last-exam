import copy
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tasks.life_sciences.idp_ensemble_scoring.scripts.cached_cspred import (
    CachedUCBShiftX,
    experimental_shifts,
)
from tasks.life_sciences.idp_ensemble_scoring.scripts.scientific_contract import (
    METHODS,
    aggregate_scores,
    normalize_columns,
)


@pytest.mark.parametrize(
    ("matrix", "expected"),
    [
        ([[0], [1], [2], [3], [4]], [0, 0.25, 0.5, 0.75, 1]),
        ([[-9], [-7], [-5], [-3], [-1]], [0, 0.25, 0.5, 0.75, 1]),
        ([[3], [3], [3], [3], [3]], [0.5] * 5),
        ([[0, 9], [1, 9], [2, 9], [3, 9], [4, 9]], [0.25, 0.375, 0.5, 0.625, 0.75]),
    ],
)
def test_column_minmax_and_neutral_degenerate_columns(matrix, expected):
    assert normalize_columns(matrix) == expected


@pytest.mark.parametrize("matrix", [[], [[]], [[1], [1, 2]], [[math.nan]], [[math.inf]]])
def test_invalid_normalization_input_is_not_silently_ignored(matrix):
    with pytest.raises(ValueError):
        normalize_columns(matrix)


@pytest.fixture
def example_scores():
    sets = {
        "CS": ["one", "two", "three"],
        "JC": ["one"],
        "NOE": ["one", "two"],
        "PRE": ["two", "three"],
    }
    generator = np.random.RandomState(27)
    records = [
        dict(
            Method=method, protein=protein, observable=label, score=float(generator.normal(-80, 20))
        )
        for method in METHODS
        for label, proteins in sets.items()
        for protein in proteins
    ]
    return sets, records


def independent_aggregation(sets, records):
    lookup = {(row["Method"], row["protein"], row["observable"]): row["score"] for row in records}
    proteins = sorted(set().union(*map(set, sets.values())))
    result = {method: {} for method in METHODS}
    for display, labels in [
        ("CS", ["CS"]),
        ("JC", ["JC"]),
        ("NOE/PRE", ["NOE", "PRE"]),
        ("Total", ["CS", "JC", "NOE", "PRE"]),
    ]:
        subset = [
            protein for protein in proteins if any(protein in sets[label] for label in labels)
        ]
        matrix = np.array(
            [
                [
                    sum(lookup.get((method, protein, label), 0) for label in labels)
                    for protein in subset
                ]
                for method in METHODS
            ]
        )
        minimum = matrix.min(axis=0)
        span = np.ptp(matrix, axis=0)
        normalized = np.full_like(matrix, 0.5)
        np.divide(matrix - minimum, span, out=normalized, where=span != 0)
        for method, value in zip(METHODS, normalized.mean(axis=1)):
            result[method][display] = value
    return result


def test_raw_union_and_total_match_separate_numpy_oracle(example_scores):
    sets, records = example_scores
    expected = independent_aggregation(sets, records)
    actual = aggregate_scores(records, sets)
    for row in actual:
        for label in ("Total", "CS", "JC", "NOE/PRE"):
            assert row[label] == pytest.approx(expected[row["Method"]][label], abs=1e-14)
    assert actual == sorted(actual, key=lambda row: (-row["Total"], row["Method"]))
    assert aggregate_scores(list(reversed(records)), sets) == actual


def test_union_combines_raw_scores_before_normalization(example_scores):
    sets, records = example_scores
    final = aggregate_scores(records, sets)
    separate = {}
    for label in ("NOE", "PRE"):
        subset = [record for record in records if record["observable"] == label]
        matrix = [
            [
                next(
                    row["score"]
                    for row in subset
                    if row["Method"] == method and row["protein"] == protein
                )
                for protein in sets[label]
            ]
            for method in METHODS
        ]
        separate[label] = normalize_columns(matrix)
    wrong = {
        method: (separate["NOE"][index] + separate["PRE"][index]) / 2
        for index, method in enumerate(METHODS)
    }
    assert any(abs(row["NOE/PRE"] - wrong[row["Method"]]) > 0.01 for row in final)
    assert any(
        abs(row["Total"] - (row["CS"] + row["JC"] + row["NOE/PRE"]) / 3) > 0.01 for row in final
    )


@pytest.mark.parametrize(
    "damage",
    ["missing", "duplicate", "extra_method", "extra_protein", "forbidden_observable", "nan", "inf"],
)
def test_missing_or_invalid_raw_scores_fail(example_scores, damage):
    sets, records = copy.deepcopy(example_scores)
    if damage == "missing":
        records.pop()
    elif damage == "duplicate":
        records.append(records[0])
    elif damage == "extra_method":
        records[0]["Method"] = "Model6"
    elif damage == "extra_protein":
        records[0]["protein"] = "unknown"
    elif damage == "forbidden_observable":
        records[0]["observable"] = "SAXS"
    else:
        records[0]["score"] = float(damage)
    with pytest.raises(ValueError):
        aggregate_scores(records, sets)


def test_all_tied_columns_have_deterministic_neutral_result(example_scores):
    sets, records = example_scores
    for record in records:
        record["score"] = -100.0
    final = aggregate_scores(records, sets)
    assert [row["Method"] for row in final] == list(METHODS)
    assert all(row[label] == 0.5 for row in final for label in ("Total", "CS", "JC", "NOE/PRE"))


def test_cached_batch_preserves_row_boundaries_and_loads_each_model_once():
    loads = []

    class Model:
        def predict(self, features):
            return features.sum(axis=1)

    def load(path):
        loads.append(path)
        return Model()

    cspred = SimpleNamespace(
        ML_MODEL_PATH="/unused",
        toolbox=SimpleNamespace(ATOMS=["H", "N"]),
        joblib=SimpleNamespace(load=load),
        sparta_rename_map={},
        rcoil_cols=["RCOIL_H", "RCOIL_N"],
        data_preprocessing=lambda frame: frame[["feature"]],
        prepare_data_for_atom=lambda frame, atom: frame.copy(),
    )
    first = pd.DataFrame(
        dict(
            RESNAME=["ALA", "GLY"],
            RES_NUM=[1, 2],
            feature=[3.0, 5.0],
            RCOIL_H=[8.0, 8.0],
            RCOIL_N=[120.0, 120.0],
        )
    )
    second = first.iloc[:1].copy()
    second["feature"] = 11.0
    predictor = CachedUCBShiftX(cspred, prediction_jobs=2)
    actual = predictor.predict_features([first, second])
    assert len(loads) == 4
    assert all(model.n_jobs == 2 for model in predictor.models.values())
    assert actual[0]["H_X"].tolist() == [14.0, 18.0]
    assert actual[1]["H_X"].tolist() == [30.0]
    again = predictor.predict_features([second, first])
    assert len(loads) == 4
    pd.testing.assert_frame_equal(again[0], actual[1])
    pd.testing.assert_frame_equal(again[1], actual[0])
    with pytest.raises(ValueError):
        predictor.predict_features([first.iloc[:0]])


def test_prediction_selection_uses_residue_identity_and_experimental_order():
    prediction = pd.DataFrame(dict(RESNUM=[8, 2], H_X=[8.2, 7.8], N_X=[120.0, 122.0]))
    experiment = pd.DataFrame(dict(resnum=[2, 8, 2], atomname=["H", "N", "N"]))
    assert experimental_shifts(prediction, experiment).tolist() == [7.8, 120.0, 122.0]


@pytest.mark.parametrize("damage", ["missing", "duplicate", "nan"])
def test_requested_prediction_corruption_fails(damage):
    prediction = pd.DataFrame(dict(RESNUM=[1, 2], H_X=[8.0, 9.0]))
    experiment = pd.DataFrame(dict(resnum=[1, 2], atomname=["H", "H"]))
    if damage == "missing":
        prediction = prediction.iloc[:1]
    elif damage == "duplicate":
        prediction["RESNUM"] = [1, 1]
    else:
        prediction.loc[0, "H_X"] = np.nan
    with pytest.raises((ValueError, KeyError)):
        experimental_shifts(prediction, experiment)


def test_authentic_new_runtime_and_batched_prediction_equivalence():
    value = os.environ.get("IDP_SCIENTIFIC_EVIDENCE")
    if not value:
        pytest.skip("Set IDP_SCIENTIFIC_EVIDENCE for retained-guest scientific evidence")
    root = Path(value) / "guest"
    benchmark = json.loads((root / "benchmark.json").read_text())
    assert benchmark["python"].startswith("3.7.12")
    for name, version in {
        "numpy": "1.19.0",
        "pandas": "1.1.0",
        "scikit-learn": "0.22",
        "biopython": "1.74",
        "joblib": "0.17.0",
        "scipy": "1.5.4",
    }.items():
        assert benchmark["packages"][name] == version
    assert len(benchmark["samples"]) == 3
    assert all(sample["requested_max_abs_error"] < 1e-10 for sample in benchmark["samples"])
    assert all(sample["max_abs_error"] < 1e-10 for sample in benchmark["samples"])
    assert all(model["n_jobs"] == 4 for model in benchmark["model_parameters"].values())
    pilot = json.loads((root / "pilot/benchmark.json").read_text())
    assert len(pilot["proteins"]) == 19
    assert pilot["feature_workers"] == 12
    assert pilot["prediction_workers"] == 4
    audit = json.loads((root / "staged-hash-audit.json").read_text())
    assert len(audit["differences"]) == 4
    assert not any(name.endswith(".sav") for name in audit["differences"])
