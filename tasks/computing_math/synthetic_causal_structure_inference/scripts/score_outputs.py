"""Deterministic scorer for synthetic_causal_structure_inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def zero_report(
    truth_map: dict[str, dict[str, Any]], *, schema_errors: list[str] | None = None
) -> dict[str, Any]:
    dataset_results = [
        {
            "dataset_id": dataset_id,
            "scenario_correct": 0.0,
            "strategy_correct": 0.0,
            "identifiable_correct": 0.0,
            "edge_f1": 0.0,
            "latent_confounder_f1": 0.0,
            "role_accuracy": 0.0,
            "combined_score": 0.0,
        }
        for dataset_id in sorted(truth_map)
    ]
    aggregate = {
        "num_datasets": len(dataset_results),
        "scenario_accuracy": 0.0,
        "strategy_accuracy": 0.0,
        "identifiable_accuracy": 0.0,
        "edge_f1_macro": 0.0,
        "latent_confounder_f1_macro": 0.0,
        "role_accuracy_macro": 0.0,
        "overall_score": 0.0,
    }
    if schema_errors:
        aggregate["schema_errors"] = schema_errors
    return {"aggregate": aggregate, "per_dataset": dataset_results}


def normalize_edge_list(edges: list[list[str]] | None) -> set[tuple[str, str]]:
    if not edges:
        return set()
    normalized: set[tuple[str, str]] = set()
    for edge in edges:
        if not isinstance(edge, list) or len(edge) != 2:
            continue
        normalized.add((str(edge[0]), str(edge[1])))
    return normalized


def normalize_pair_list(pairs: list[list[str]] | None) -> set[tuple[str, str]]:
    if not pairs:
        return set()
    normalized: set[tuple[str, str]] = set()
    for pair in pairs:
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        normalized.add(tuple(sorted((str(pair[0]), str(pair[1])))))
    return normalized


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def f1_from_sets(predicted: set[tuple[str, str]], truth: set[tuple[str, str]]) -> float:
    if not predicted and not truth:
        return 1.0
    true_positive = len(predicted & truth)
    precision = safe_div(true_positive, len(predicted))
    recall = safe_div(true_positive, len(truth))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def role_accuracy(predicted_roles: dict[str, str], truth_roles: dict[str, str]) -> float:
    if not truth_roles:
        return 1.0
    correct = 0
    for variable, truth_role in truth_roles.items():
        if predicted_roles.get(variable) == truth_role:
            correct += 1
    return correct / len(truth_roles)


def validate_submission_schema(
    submission: dict[str, Any], reference: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]] | None, list[str]]:
    truth_map = {entry["dataset_id"]: entry for entry in reference.get("predictions", [])}
    errors: list[str] = []

    if not isinstance(submission, dict):
        return truth_map, None, ["submission root must be a JSON object"]

    expected_keys = {"benchmark_name", "predictions"}
    actual_keys = set(submission)
    if actual_keys != expected_keys:
        errors.append(
            "submission top-level keys must be exactly "
            f"{sorted(expected_keys)}; got {sorted(actual_keys)}"
        )

    expected_benchmark_name = reference.get("benchmark_name")
    benchmark_name = submission.get("benchmark_name")
    if not isinstance(benchmark_name, str):
        errors.append("benchmark_name must be a string")
    elif isinstance(expected_benchmark_name, str) and benchmark_name != expected_benchmark_name:
        errors.append(
            "benchmark_name must match the benchmark manifest/reference value "
            f"{expected_benchmark_name!r}"
        )

    predictions = submission.get("predictions")
    if not isinstance(predictions, list):
        errors.append("predictions must be a list")
        return truth_map, None, errors

    dataset_ids: list[str] = []
    for index, entry in enumerate(predictions):
        if not isinstance(entry, dict):
            errors.append(f"prediction at index {index} must be an object")
            continue
        dataset_id = entry.get("dataset_id")
        if not isinstance(dataset_id, str):
            errors.append(f"prediction at index {index} must include a string dataset_id")
            continue
        dataset_ids.append(dataset_id)

    if len(dataset_ids) != len(set(dataset_ids)):
        errors.append("predictions must not contain duplicate dataset_id values")

    truth_ids = set(truth_map)
    predicted_ids = set(dataset_ids)
    if predicted_ids != truth_ids:
        missing_ids = sorted(truth_ids - predicted_ids)
        extra_ids = sorted(predicted_ids - truth_ids)
        if missing_ids:
            errors.append(f"predictions missing dataset ids: {missing_ids}")
        if extra_ids:
            errors.append(f"predictions contain unknown dataset ids: {extra_ids}")

    return truth_map, predictions, errors


def evaluate_submission_json(submission: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    truth_map, prediction_entries, schema_errors = validate_submission_schema(submission, reference)
    if schema_errors:
        return zero_report(truth_map, schema_errors=schema_errors)

    assert prediction_entries is not None
    predictions = {entry["dataset_id"]: entry for entry in prediction_entries}

    dataset_results: list[dict[str, Any]] = []
    for dataset_id, truth in sorted(truth_map.items()):
        predicted = predictions.get(dataset_id, {})
        scenario_correct = float(predicted.get("scenario") == truth["scenario"])
        strategy_correct = float(
            predicted.get("identification_strategy") == truth["identification_strategy"]
        )
        identifiable_correct = float(
            predicted.get("identifiable_effect") == truth["identifiable_effect"]
        )
        edge_f1 = f1_from_sets(
            normalize_edge_list(predicted.get("directed_edges")),
            normalize_edge_list(truth.get("directed_edges")),
        )
        latent_f1 = f1_from_sets(
            normalize_pair_list(predicted.get("latent_confounders")),
            normalize_pair_list(truth.get("latent_confounders")),
        )
        roles_score = role_accuracy(
            predicted.get("variable_roles", {})
            if isinstance(predicted.get("variable_roles", {}), dict)
            else {},
            truth.get("variable_roles", {}),
        )
        combined = (
            0.30 * scenario_correct
            + 0.20 * strategy_correct
            + 0.10 * identifiable_correct
            + 0.20 * edge_f1
            + 0.10 * latent_f1
            + 0.10 * roles_score
        )
        dataset_results.append(
            {
                "dataset_id": dataset_id,
                "scenario_correct": scenario_correct,
                "strategy_correct": strategy_correct,
                "identifiable_correct": identifiable_correct,
                "edge_f1": edge_f1,
                "latent_confounder_f1": latent_f1,
                "role_accuracy": roles_score,
                "combined_score": combined,
            }
        )

    aggregate = {
        "num_datasets": len(dataset_results),
        "scenario_accuracy": safe_div(
            sum(item["scenario_correct"] for item in dataset_results),
            len(dataset_results),
        ),
        "strategy_accuracy": safe_div(
            sum(item["strategy_correct"] for item in dataset_results),
            len(dataset_results),
        ),
        "identifiable_accuracy": safe_div(
            sum(item["identifiable_correct"] for item in dataset_results),
            len(dataset_results),
        ),
        "edge_f1_macro": safe_div(
            sum(item["edge_f1"] for item in dataset_results),
            len(dataset_results),
        ),
        "latent_confounder_f1_macro": safe_div(
            sum(item["latent_confounder_f1"] for item in dataset_results),
            len(dataset_results),
        ),
        "role_accuracy_macro": safe_div(
            sum(item["role_accuracy"] for item in dataset_results),
            len(dataset_results),
        ),
        "overall_score": safe_div(
            sum(item["combined_score"] for item in dataset_results),
            len(dataset_results),
        ),
    }
    return {"aggregate": aggregate, "per_dataset": dataset_results}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_submission_files(submission_path: Path, reference_path: Path) -> dict[str, Any]:
    return evaluate_submission_json(load_json(submission_path), load_json(reference_path))
