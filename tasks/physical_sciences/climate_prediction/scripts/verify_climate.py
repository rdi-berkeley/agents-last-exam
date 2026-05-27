"""VM-side verifier for climate_prediction benchmark.

Reads agent output, compares against hidden reference truth, and prints
a JSON verdict to stdout.  Debug diagnostics go to stderr.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


REQUIRED_FILES = [
    "processed/train_inputs.npy",
    "processed/train_outputs.npy",
    "processed/test_inputs.npy",
    "processed/metadata.json",
    "processed/test_predictions.npy",
    "submissions/kaggle_submission.csv",
]

EXPECTED_PRED_SHAPE = (120, 2, 48, 72)
EXPECTED_CSV_ROWS = 829440


def _load_npy(path):
    return np.load(path)


def _check_files(output_dir):
    missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(output_dir, f))]
    return missing


def _check_shapes(output_dir, expected_shapes):
    errors = []
    for name, expected in expected_shapes.items():
        path = os.path.join(output_dir, f"processed/{name}.npy")
        if not os.path.exists(path):
            continue
        arr = _load_npy(path)
        if list(arr.shape) != expected:
            errors.append(f"{name}: got {list(arr.shape)}, expected {expected}")
    return errors


def _check_csv(output_dir):
    csv_path = os.path.join(output_dir, "submissions/kaggle_submission.csv")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return False, str(e)
    if len(df) != EXPECTED_CSV_ROWS:
        return False, f"row count {len(df)} != {EXPECTED_CSV_ROWS}"
    if "ID" not in df.columns or "Prediction" not in df.columns:
        return False, f"missing columns: {list(df.columns)}"
    return True, "ok"


def _compute_metrics(preds, truth, train_outputs):
    tas_rmse = float(np.sqrt(np.mean((preds[:, 0] - truth[:, 0]) ** 2)))
    pr_rmse = float(np.sqrt(np.mean((preds[:, 1] - truth[:, 1]) ** 2)))

    tas_mean_map_rmse = float(
        np.sqrt(np.mean((preds[:, 0].mean(axis=0) - truth[:, 0].mean(axis=0)) ** 2))
    )
    pr_mean_map_rmse = float(
        np.sqrt(np.mean((preds[:, 1].mean(axis=0) - truth[:, 1].mean(axis=0)) ** 2))
    )

    clim = train_outputs.mean(axis=0)
    clim_tiled = np.tile(clim, (truth.shape[0], 1, 1, 1))
    clim_rmse = float(np.sqrt(np.mean((clim_tiled - truth) ** 2)))
    pred_rmse = float(np.sqrt(np.mean((preds - truth) ** 2)))
    skill = 1.0 - pred_rmse / clim_rmse if clim_rmse > 0 else 0.0

    return {
        "tas_rmse": tas_rmse,
        "pr_rmse": pr_rmse,
        "tas_mean_map_rmse": tas_mean_map_rmse,
        "pr_mean_map_rmse": pr_mean_map_rmse,
        "skill": skill,
        "clim_rmse": clim_rmse,
        "pred_rmse": pred_rmse,
    }


def verify(output_dir, reference_dir, input_dir):
    result = {"score": 0.0}

    missing = _check_files(output_dir)
    if missing:
        result["detail"] = f"missing files: {missing}"
        return result

    with open(os.path.join(input_dir, "metadata.json")) as f:
        meta = json.load(f)

    expected_shapes = {
        "train_inputs": meta["shapes"]["train_inputs"],
        "train_outputs": meta["shapes"]["train_outputs"],
        "test_inputs": meta["shapes"]["test_inputs"],
        "test_predictions": list(EXPECTED_PRED_SHAPE),
    }

    shape_errors = _check_shapes(output_dir, expected_shapes)
    if shape_errors:
        result["detail"] = f"shape errors: {shape_errors}"
        return result

    csv_ok, csv_detail = _check_csv(output_dir)

    preds = _load_npy(os.path.join(output_dir, "processed/test_predictions.npy"))
    truth = _load_npy(os.path.join(reference_dir, "processed/test_outputs.npy"))
    train_out = _load_npy(os.path.join(output_dir, "processed/train_outputs.npy"))

    if preds.shape != truth.shape:
        result["detail"] = f"prediction shape {list(preds.shape)} != truth {list(truth.shape)}"
        return result

    metrics = _compute_metrics(preds, truth, train_out)
    result.update(metrics)

    with open(os.path.join(reference_dir, "evaluation_contract.json")) as f:
        thresholds = json.load(f)["hidden_accuracy_thresholds"]

    score = 0.0

    if csv_ok:
        score += 0.05
    result["csv_ok"] = csv_ok
    result["csv_detail"] = csv_detail

    if metrics["tas_rmse"] <= thresholds["tas_monthly_rmse_max"]:
        score += 0.10
    if metrics["tas_mean_map_rmse"] <= thresholds["tas_mean_map_rmse_max"]:
        score += 0.10
    if metrics["pr_rmse"] <= thresholds["pr_monthly_rmse_max"]:
        score += 0.10
    if metrics["pr_mean_map_rmse"] <= thresholds["pr_mean_map_rmse_max"]:
        score += 0.10

    skill_contribution = max(0.0, min(1.0, metrics["skill"])) * 0.55
    score += skill_contribution

    result["score"] = round(min(score, 1.0), 4)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args()

    try:
        result = verify(args.output_dir, args.reference_dir, args.input_dir)
    except Exception as e:
        result = {"score": 0.0, "error": str(e)}
        print(json.dumps(result), file=sys.stderr)
        print(json.dumps(result))
        sys.exit(0)

    print(json.dumps(result, indent=2), file=sys.stderr)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
