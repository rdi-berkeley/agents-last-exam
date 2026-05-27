#!/usr/bin/env python
"""Local scorer for CTQW policy transfer graph classification outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PREDICTION_FIELDS = ["graph_id", "predicted_label", "score"]
POLICY_FILE = "selected_observation_policy.json"
SOURCE_RESULTS_FILE = "source_results.tsv"
SOURCE_PREDICTIONS_FILE = "source_predictions.tsv"
TARGET_RESULTS_FILE = "target_results.tsv"
TARGET_PREDICTIONS_FILE = "target_predictions.tsv"
RUN_CONFIG_FILE = "run_config.json"
REPORT_FILE = "report.md"
REQUIRED_OUTPUT_FILES = [
    POLICY_FILE,
    SOURCE_RESULTS_FILE,
    SOURCE_PREDICTIONS_FILE,
    TARGET_RESULTS_FILE,
    TARGET_PREDICTIONS_FILE,
    RUN_CONFIG_FILE,
    REPORT_FILE,
]


@dataclass
class ScoreResult:
    score: float
    passed: bool
    source_accuracy: float
    target_accuracy: float
    threshold: float
    reasons: list[str]


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _read_tsv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        rows = list(reader)
    return list(reader.fieldnames or []), rows


def _load_truth_map(manifest_path: Path, *, split: str | None = None) -> dict[str, str]:
    _, rows = _read_tsv_rows(manifest_path)
    truth: dict[str, str] = {}
    for row in rows:
        if split is not None and row.get("split") != split:
            continue
        graph_id = row.get("graph_id", "").strip()
        label = row.get("label", "").strip()
        if not graph_id or not label:
            continue
        truth[graph_id] = label
    return truth


def _fail(reasons: list[str], message: str) -> None:
    reasons.append(message)


def _parse_float(raw: str, *, label: str, reasons: list[str]) -> float | None:
    try:
        return float(raw)
    except (TypeError, ValueError):
        _fail(reasons, f"{label} is not a valid float: {raw!r}")
        return None


def _parse_int(raw: Any, *, label: str, reasons: list[str]) -> int | None:
    if isinstance(raw, bool):
        _fail(reasons, f"{label} must be an integer, got boolean")
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        _fail(reasons, f"{label} is not a valid integer: {raw!r}")
        return None


def _approx_equal(left: float, right: float, tol: float = 1e-4) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tol)


def _validate_observable(observable: str, max_qubits: int) -> int | None:
    parts = observable.split()
    if not parts:
        return None
    seen_qubits: set[int] = set()
    for part in parts:
        match = re.fullmatch(r"([XYZ])(\d+)", part)
        if match is None:
            return None
        qubit = int(match.group(2))
        if qubit < 0 or qubit >= max_qubits or qubit in seen_qubits:
            return None
        seen_qubits.add(qubit)
    return len(parts)


def _read_predictions(
    path: Path,
    *,
    expected_truth: dict[str, str],
    label: str,
    reasons: list[str],
) -> tuple[dict[str, str], dict[str, float]] | None:
    fields, rows = _read_tsv_rows(path)
    if fields != PREDICTION_FIELDS:
        _fail(reasons, f"{label} header must be exactly {PREDICTION_FIELDS}, got {fields}")
        return None

    predictions: dict[str, str] = {}
    scores: dict[str, float] = {}
    allowed_labels = set(expected_truth.values())
    expected_ids = set(expected_truth)

    for row in rows:
        graph_id = row["graph_id"].strip()
        predicted_label = row["predicted_label"].strip()
        if graph_id in predictions:
            _fail(reasons, f"{label} contains duplicate graph_id {graph_id!r}")
            continue
        if graph_id not in expected_ids:
            _fail(reasons, f"{label} contains unexpected graph_id {graph_id!r}")
            continue
        if predicted_label not in allowed_labels:
            _fail(reasons, f"{label} uses unknown class {predicted_label!r} for {graph_id}")
            continue
        score = _parse_float(row["score"], label=f"{label} score for {graph_id}", reasons=reasons)
        if score is None:
            continue
        if score < 0.0 or score > 1.0:
            _fail(reasons, f"{label} score for {graph_id} must be within [0, 1]")
            continue
        predictions[graph_id] = predicted_label
        scores[graph_id] = score

    missing = sorted(expected_ids - set(predictions))
    extra = sorted(set(predictions) - expected_ids)
    if missing:
        _fail(reasons, f"{label} is missing graph_ids: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if extra:
        _fail(reasons, f"{label} includes extra graph_ids: {extra[:5]}{'...' if len(extra) > 5 else ''}")

    if reasons:
        return None
    return predictions, scores


def _accuracy(predictions: dict[str, str], truth: dict[str, str]) -> float:
    correct = sum(1 for graph_id, label in truth.items() if predictions.get(graph_id) == label)
    return correct / len(truth)


def _read_metric_table(path: Path) -> dict[str, str] | None:
    fields, rows = _read_tsv_rows(path)
    normalized_fields = [field.strip().lower() for field in fields]
    if normalized_fields == ["metric", "value"]:
        return {row["metric"].strip().lower(): row["value"].strip() for row in rows}
    if len(rows) == 1:
        return {field.strip().lower(): rows[0][field].strip() for field in fields}
    return None


def _extract_report_metrics(report_text: str) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    current_section = ""
    for line in report_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            current_section = stripped[3:].strip().lower()
            metrics.setdefault(current_section, {})
            continue
        if not stripped.startswith("-"):
            continue
        match = re.match(r"-\s+([^:]+):\s*([0-9.]+)", stripped)
        if match is None or not current_section:
            continue
        key = match.group(1).strip().lower()
        try:
            value = float(match.group(2))
        except ValueError:
            continue
        metrics.setdefault(current_section, {})[key] = value
    return metrics


def _report_confirms_no_retraining(report_text: str) -> bool:
    for line in report_text.lower().splitlines():
        if "retrain" not in line:
            continue
        if "no " in line or "not " in line or "without " in line:
            return True
    return False


def _validate_policy_and_run_config(
    *,
    input_dir: Path,
    agent_output_dir: Path,
    reasons: list[str],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    policy = _read_json(agent_output_dir / POLICY_FILE)
    run_config = _read_json(agent_output_dir / RUN_CONFIG_FILE)
    measurement_config = _read_json(input_dir / "measurement_config.json")
    ctqw_config = _read_json(input_dir / "ctqw_config.json")
    observable_pool = _read_json(input_dir / "observable_pool.json")

    if policy.get("policy_type") != "fixed":
        _fail(reasons, "selected_observation_policy.json must declare policy_type='fixed'")

    evolution_time = policy.get("ctqw_parameters", {}).get("evolution_time")
    evolution_time_value = _parse_float(
        evolution_time,
        label="selected_observation_policy.ctqw_parameters.evolution_time",
        reasons=reasons,
    )
    min_time = _parse_float(
        ctqw_config.get("time_range", {}).get("min"),
        label="input/ctqw_config.json time_range.min",
        reasons=reasons,
    )
    max_time = _parse_float(
        ctqw_config.get("time_range", {}).get("max"),
        label="input/ctqw_config.json time_range.max",
        reasons=reasons,
    )
    if (
        evolution_time_value is not None
        and min_time is not None
        and max_time is not None
        and not (min_time <= evolution_time_value <= max_time)
    ):
        _fail(reasons, "selected_observation_policy evolution_time is outside input/ctqw_config.json time_range")

    observable_order = _parse_int(
        policy.get("observable_order"),
        label="selected_observation_policy.observable_order",
        reasons=reasons,
    )
    order_min = _parse_int(
        measurement_config.get("observable_order", {}).get("min"),
        label="input/measurement_config.json observable_order.min",
        reasons=reasons,
    )
    order_max = _parse_int(
        measurement_config.get("observable_order", {}).get("max"),
        label="input/measurement_config.json observable_order.max",
        reasons=reasons,
    )
    pool_max_order = _parse_int(
        observable_pool.get("max_order"),
        label="input/observable_pool.json max_order",
        reasons=reasons,
    )
    if (
        observable_order is not None
        and order_min is not None
        and order_max is not None
        and not (order_min <= observable_order <= order_max)
    ):
        _fail(reasons, "selected_observation_policy observable_order is outside measurement_config bounds")
    if observable_order is not None and pool_max_order is not None and observable_order > pool_max_order:
        _fail(reasons, "selected_observation_policy observable_order exceeds observable_pool max_order")

    selected_observables = policy.get("selected_observables")
    if not isinstance(selected_observables, list) or not selected_observables:
        _fail(reasons, "selected_observation_policy.selected_observables must be a non-empty list")
        selected_observables = []
    elif len(selected_observables) != len(set(selected_observables)):
        _fail(reasons, "selected_observation_policy.selected_observables contains duplicates")

    num_qubits = _parse_int(
        observable_pool.get("num_qubits"),
        label="input/observable_pool.json num_qubits",
        reasons=reasons,
    )
    feature_dim_budget = _parse_int(
        measurement_config.get("feature_dim_budget"),
        label="input/measurement_config.json feature_dim_budget",
        reasons=reasons,
    )
    if feature_dim_budget is not None and len(selected_observables) > feature_dim_budget:
        _fail(reasons, "selected_observation_policy selects more observables than feature_dim_budget allows")

    if num_qubits is not None and observable_order is not None:
        for observable in selected_observables:
            actual_order = _validate_observable(str(observable), num_qubits)
            if actual_order is None:
                _fail(reasons, f"Invalid observable syntax in selected_observation_policy: {observable!r}")
                continue
            if actual_order > observable_order:
                _fail(reasons, f"Observable {observable!r} exceeds declared observable_order={observable_order}")

    shot_allocation = policy.get("shot_allocation")
    if not isinstance(shot_allocation, dict) or not shot_allocation:
        _fail(reasons, "selected_observation_policy.shot_allocation must be a non-empty object")
        shot_allocation = {}
    if set(shot_allocation) != set(selected_observables):
        _fail(reasons, "shot_allocation keys must match selected_observables exactly")

    total_allocated = 0
    for observable, shot_count in shot_allocation.items():
        shot_value = _parse_int(
            shot_count,
            label=f"selected_observation_policy.shot_allocation[{observable!r}]",
            reasons=reasons,
        )
        if shot_value is None:
            continue
        if shot_value <= 0:
            _fail(reasons, f"selected_observation_policy.shot_allocation[{observable!r}] must be positive")
            continue
        total_allocated += shot_value

    total_shots = _parse_int(
        policy.get("total_shots"),
        label="selected_observation_policy.total_shots",
        reasons=reasons,
    )
    shot_budget = _parse_int(
        measurement_config.get("shot_budget"),
        label="input/measurement_config.json shot_budget",
        reasons=reasons,
    )
    if total_shots is not None and total_allocated and total_allocated != total_shots:
        _fail(reasons, "selected_observation_policy.total_shots does not equal the sum of shot_allocation")
    if total_shots is not None and shot_budget is not None and total_shots > shot_budget:
        _fail(reasons, "selected_observation_policy.total_shots exceeds input/measurement_config.json shot_budget")

    run_shot_budget = _parse_int(
        run_config.get("shot_budget"),
        label="run_config.shot_budget",
        reasons=reasons,
    )
    if total_shots is not None and run_shot_budget is not None and total_shots != run_shot_budget:
        _fail(reasons, "run_config.shot_budget must match selected_observation_policy.total_shots")

    random_seed = _parse_int(run_config.get("random_seed"), label="run_config.random_seed", reasons=reasons)
    num_trials = _parse_int(run_config.get("num_trials"), label="run_config.num_trials", reasons=reasons)
    if random_seed is not None and random_seed < 0:
        _fail(reasons, "run_config.random_seed must be non-negative")
    if num_trials is not None and num_trials <= 0:
        _fail(reasons, "run_config.num_trials must be positive")

    classifier = run_config.get("classifier")
    if not isinstance(classifier, dict):
        _fail(reasons, "run_config.classifier must be an object")
    else:
        if classifier.get("type") != "logistic_regression":
            _fail(reasons, "run_config.classifier.type must be 'logistic_regression'")
        max_iter = _parse_int(
            classifier.get("max_iter"),
            label="run_config.classifier.max_iter",
            reasons=reasons,
        )
        if max_iter is not None and max_iter <= 0:
            _fail(reasons, "run_config.classifier.max_iter must be positive")

    if not isinstance(run_config.get("normalize_features"), bool):
        _fail(reasons, "run_config.normalize_features must be a boolean")

    return policy, run_config


def _validate_source_results(
    path: Path,
    *,
    policy: dict[str, Any] | None,
    source_accuracy: float,
    reasons: list[str],
) -> None:
    fields, rows = _read_tsv_rows(path)
    if not fields or not rows:
        _fail(reasons, "source_results.tsv must be a non-empty TSV")
        return

    row = rows[0]
    if policy is None:
        return

    if "evolution_time" in row:
        value = _parse_float(row["evolution_time"], label="source_results.tsv evolution_time", reasons=reasons)
        expected = policy.get("ctqw_parameters", {}).get("evolution_time")
        expected_value = _parse_float(
            expected,
            label="selected_observation_policy.ctqw_parameters.evolution_time",
            reasons=reasons,
        )
        if value is not None and expected_value is not None and not _approx_equal(value, expected_value):
            _fail(reasons, "source_results.tsv evolution_time disagrees with selected_observation_policy.json")

    if "num_observables" in row:
        value = _parse_int(row["num_observables"], label="source_results.tsv num_observables", reasons=reasons)
        expected = len(policy.get("selected_observables", []))
        if value is not None and value != expected:
            _fail(reasons, "source_results.tsv num_observables disagrees with selected_observation_policy.json")

    if "test_accuracy" in row:
        value = _parse_float(row["test_accuracy"], label="source_results.tsv test_accuracy", reasons=reasons)
        if value is not None and not _approx_equal(value, source_accuracy, tol=1e-4):
            _fail(reasons, "source_results.tsv test_accuracy disagrees with computed source accuracy")


def _validate_target_results(
    path: Path,
    *,
    target_accuracy: float,
    expected_num_graphs: int,
    reasons: list[str],
) -> None:
    metrics = _read_metric_table(path)
    if metrics is None:
        _fail(reasons, "target_results.tsv must be either a metric/value table or a one-row TSV with named columns")
        return

    accuracy_raw = metrics.get("accuracy")
    if accuracy_raw is not None:
        accuracy = _parse_float(accuracy_raw, label="target_results.tsv accuracy", reasons=reasons)
        if accuracy is not None and not _approx_equal(accuracy, target_accuracy, tol=1e-4):
            _fail(reasons, "target_results.tsv accuracy disagrees with computed target accuracy")

    num_graphs_raw = metrics.get("num_graphs")
    if num_graphs_raw is not None:
        num_graphs = _parse_int(num_graphs_raw, label="target_results.tsv num_graphs", reasons=reasons)
        if num_graphs is not None and num_graphs != expected_num_graphs:
            _fail(reasons, "target_results.tsv num_graphs disagrees with target manifest size")


def _validate_report(
    path: Path,
    *,
    source_accuracy: float,
    target_accuracy: float,
    reasons: list[str],
) -> None:
    report_text = path.read_text(encoding="utf-8")
    metrics = _extract_report_metrics(report_text)

    target_section = metrics.get("target task performance", {})
    report_target_accuracy = target_section.get("accuracy")
    if report_target_accuracy is not None and not _approx_equal(report_target_accuracy, target_accuracy, tol=5e-4):
        _fail(reasons, "report.md target accuracy disagrees with computed target accuracy")

    transfer_gap_section = metrics.get("transfer gap", {})
    report_source_accuracy = transfer_gap_section.get("source test accuracy")
    if report_source_accuracy is not None and not _approx_equal(report_source_accuracy, source_accuracy, tol=5e-4):
        _fail(reasons, "report.md source test accuracy disagrees with computed source accuracy")

    if not _report_confirms_no_retraining(report_text):
        _fail(reasons, "report.md must explicitly confirm that no target-side retraining was performed")


def score_submission_dir(
    *,
    agent_output_dir: Path,
    input_dir: Path,
    source_truth_manifest: Path,
    target_truth_manifest: Path,
) -> ScoreResult:
    reasons: list[str] = []

    for name in REQUIRED_OUTPUT_FILES:
        if not (agent_output_dir / name).is_file():
            _fail(reasons, f"Missing required output file: {name}")

    if reasons:
        return ScoreResult(
            score=0.0,
            passed=False,
            source_accuracy=0.0,
            target_accuracy=0.0,
            threshold=0.0,
            reasons=reasons,
        )

    eval_config = _read_json(input_dir / "eval_config.json")
    threshold = float(eval_config["target_accuracy_threshold"])

    source_truth = _load_truth_map(source_truth_manifest, split="test")
    target_truth = _load_truth_map(target_truth_manifest, split="test")
    if not source_truth:
        _fail(reasons, "Hidden source truth manifest has no labeled test rows")
    if not target_truth:
        _fail(reasons, "Hidden target truth manifest has no labeled test rows")
    if reasons:
        return ScoreResult(
            score=0.0,
            passed=False,
            source_accuracy=0.0,
            target_accuracy=0.0,
            threshold=threshold,
            reasons=reasons,
        )

    source_data = _read_predictions(
        agent_output_dir / SOURCE_PREDICTIONS_FILE,
        expected_truth=source_truth,
        label=SOURCE_PREDICTIONS_FILE,
        reasons=reasons,
    )
    target_data = _read_predictions(
        agent_output_dir / TARGET_PREDICTIONS_FILE,
        expected_truth=target_truth,
        label=TARGET_PREDICTIONS_FILE,
        reasons=reasons,
    )
    if source_data is None or target_data is None:
        return ScoreResult(
            score=0.0,
            passed=False,
            source_accuracy=0.0,
            target_accuracy=0.0,
            threshold=threshold,
            reasons=reasons,
        )

    source_predictions, _ = source_data
    target_predictions, _ = target_data
    source_accuracy = _accuracy(source_predictions, source_truth)
    target_accuracy = _accuracy(target_predictions, target_truth)

    policy, _ = _validate_policy_and_run_config(
        input_dir=input_dir,
        agent_output_dir=agent_output_dir,
        reasons=reasons,
    )
    _validate_source_results(
        agent_output_dir / SOURCE_RESULTS_FILE,
        policy=policy,
        source_accuracy=source_accuracy,
        reasons=reasons,
    )
    _validate_target_results(
        agent_output_dir / TARGET_RESULTS_FILE,
        target_accuracy=target_accuracy,
        expected_num_graphs=len(target_truth),
        reasons=reasons,
    )
    _validate_report(
        agent_output_dir / REPORT_FILE,
        source_accuracy=source_accuracy,
        target_accuracy=target_accuracy,
        reasons=reasons,
    )

    if target_accuracy + 1e-9 < threshold:
        _fail(
            reasons,
            f"target accuracy {target_accuracy:.6f} is below threshold {threshold:.6f}",
        )

    passed = not reasons
    return ScoreResult(
        score=1.0 if passed else 0.0,
        passed=passed,
        source_accuracy=source_accuracy,
        target_accuracy=target_accuracy,
        threshold=threshold,
        reasons=reasons,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-dir", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--target-manifest", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = score_submission_dir(
        agent_output_dir=args.agent_dir,
        input_dir=args.input_dir,
        source_truth_manifest=args.source_manifest,
        target_truth_manifest=args.target_manifest,
    )
    print(json.dumps(asdict(result), indent=2))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
