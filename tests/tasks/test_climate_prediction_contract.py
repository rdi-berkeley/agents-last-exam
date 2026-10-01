from __future__ import annotations

import importlib
import json

import pytest

np = pytest.importorskip("numpy")

verify_climate = importlib.import_module(
    "tasks.physical_sciences.climate_prediction.scripts.verify_climate"
)
_compute_metrics = verify_climate._compute_metrics


def test_climatology_baseline_uses_trusted_reference_units() -> None:
    reference_train = np.array(
        [[[[280.0]], [[2.0]]], [[[282.0]], [[4.0]]]], dtype=float
    )
    truth = np.array([[[[281.0]], [[3.0]]], [[[283.0]], [[5.0]]]], dtype=float)
    predictions = np.array(
        [[[[280.5]], [[2.5]]], [[[282.5]], [[4.5]]]], dtype=float
    )

    metrics = _compute_metrics(predictions, truth, reference_train)

    assert metrics["clim_rmse"] == pytest.approx(2**0.5)
    assert metrics["skill"] == pytest.approx(1 - 0.5 / (2**0.5))


def test_verify_baseline_is_independent_of_submitted_training_normalization(tmp_path, monkeypatch):
    output = tmp_path / "output"
    reference = tmp_path / "reference"
    inputs = tmp_path / "input"
    for directory in (output / "processed", output / "submissions", reference / "processed", inputs):
        directory.mkdir(parents=True)
    shape = (2, 2, 1, 1)
    monkeypatch.setattr(verify_climate, "EXPECTED_PRED_SHAPE", shape)
    monkeypatch.setattr(verify_climate, "EXPECTED_CSV_ROWS", 4)
    reference_train = np.array([[[[280.0]], [[2.0]]], [[[282.0]], [[4.0]]]])
    truth = np.array([[[[281.0]], [[3.0]]], [[[283.0]], [[5.0]]]])
    predictions = truth - 0.5
    for name in ("train_inputs", "train_outputs", "test_inputs"):
        np.save(output / "processed" / f"{name}.npy", np.zeros(shape))
    np.save(output / "processed/test_predictions.npy", predictions)
    np.save(reference / "processed/test_outputs.npy", truth)
    np.save(reference / "processed/train_outputs.npy", reference_train)
    (output / "processed/metadata.json").write_text("{}")
    (inputs / "metadata.json").write_text(json.dumps({
        "shapes": {name: list(shape) for name in ("train_inputs", "train_outputs", "test_inputs")},
    }))
    (reference / "evaluation_contract.json").write_text(json.dumps({
        "hidden_accuracy_thresholds": {
            "tas_monthly_rmse_max": 1.0,
            "tas_mean_map_rmse_max": 1.0,
            "pr_monthly_rmse_max": 1.0,
            "pr_mean_map_rmse_max": 1.0,
        },
    }))
    (output / "submissions/kaggle_submission.csv").write_text("ID,Prediction\n0,280.5\n1,2.5\n2,282.5\n3,4.5\n")
    standardized = verify_climate.verify(str(output), str(reference), str(inputs))
    np.save(output / "processed/train_outputs.npy", reference_train)
    physical_units = verify_climate.verify(str(output), str(reference), str(inputs))

    assert standardized == physical_units
    assert standardized["clim_rmse"] == pytest.approx(2**0.5)
    assert standardized["score"] == round(0.45 + 0.55 * (1 - 0.5 / 2**0.5), 4)
