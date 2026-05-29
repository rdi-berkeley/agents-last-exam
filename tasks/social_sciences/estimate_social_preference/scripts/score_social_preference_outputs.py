#!/usr/bin/env python
"""Score outputs for the estimate_social_preference task."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


VISIBLE_IDS = (
    "dataset_01",
    "dataset_02",
    "dataset_03",
    "dataset_04",
    "dataset_05",
    "dataset_06",
)
ANSWER_COMPARE_ATOL = 1e-8
ANSWER_COMPARE_RTOL = 1e-8
EPSILON = 1e-15
DEFAULT_TIMEOUT_SEC = 900


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def as_answers(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get("answers"), dict):
        return payload["answers"]
    if isinstance(payload, dict):
        return payload
    raise ValueError("answer file must be a JSON object")


def as_vector(value: Any, *, dataset_id: str, dim: int) -> list[float]:
    if not isinstance(value, list):
        raise ValueError(f"{dataset_id}: answer must be a JSON list")
    vector = [float(item) for item in value]
    if len(vector) != dim:
        raise ValueError(f"{dataset_id}: expected vector length {dim}, got {len(vector)}")
    if not all(math.isfinite(item) for item in vector):
        raise ValueError(f"{dataset_id}: vector contains non-finite values")
    return vector


def allclose(left: list[float], right: list[float]) -> bool:
    if len(left) != len(right):
        return False
    return all(
        abs(a - b) <= ANSWER_COMPARE_ATOL + ANSWER_COMPARE_RTOL * abs(b)
        for a, b in zip(left, right, strict=True)
    )


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def squared_l2(left: list[float], right: list[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right, strict=True))


def cosine_similarity(left: list[float], right: list[float]) -> float:
    denom = math.sqrt(dot(left, left)) * math.sqrt(dot(right, right))
    if denom <= 0.0:
        return 0.0
    return dot(left, right) / denom


def dataset_loss(dataset_id: str, spec: dict[str, Any], estimate: list[float]) -> dict[str, Any]:
    dim = int(spec["dim"])
    target = as_vector(spec["true_average_feature"], dataset_id=dataset_id, dim=dim)
    estimate = as_vector(estimate, dataset_id=dataset_id, dim=dim)
    sq = squared_l2(estimate, target)
    cos = cosine_similarity(estimate, target)
    metric = str(spec.get("metric", "squared_l2_error"))
    if metric == "squared_l2_error":
        loss = sq
        metric_value = sq
    elif metric == "cosine_similarity":
        loss = 1.0 - cos
        metric_value = cos
    else:
        raise ValueError(f"{dataset_id}: unsupported metric {metric!r}")
    return {
        "dataset_id": dataset_id,
        "metric": metric,
        "metric_value": float(metric_value),
        "loss": float(loss),
        "squared_l2_error": float(sq),
        "l2_error": float(math.sqrt(sq)),
        "cosine_similarity": float(cos),
        "estimate": estimate,
        "target": target,
    }


def score_answers(
    answers: dict[str, Any],
    *,
    answer_key: dict[str, Any],
    dataset_ids: list[str],
) -> dict[str, Any]:
    missing = sorted(set(dataset_ids) - set(answers))
    if missing:
        raise ValueError(f"answer file is missing dataset IDs: {missing}")
    reports = []
    for dataset_id in dataset_ids:
        spec = answer_key[dataset_id]
        reports.append(dataset_loss(dataset_id, spec, answers[dataset_id]))
    return {
        "raw_loss": float(sum(item["loss"] for item in reports) / len(reports)),
        "datasets": reports,
    }


def normalized_score(expert_loss: float, candidate_loss: float) -> float:
    if expert_loss == 0.0 and candidate_loss == 0.0:
        return 1.0
    return min(1.0, float(expert_loss) / max(float(candidate_loss), EPSILON))


def summarize_against_expert(candidate: dict[str, Any], expert: dict[str, Any], *, source: str) -> dict[str, Any]:
    expert_by_id = {item["dataset_id"]: item for item in expert["datasets"]}
    datasets = []
    for item in candidate["datasets"]:
        enriched = dict(item)
        expert_loss = float(expert_by_id[item["dataset_id"]]["loss"])
        enriched["expert_loss"] = expert_loss
        enriched["normalized_score"] = normalized_score(expert_loss, float(item["loss"]))
        datasets.append(enriched)
    return {
        "source": source,
        "score": float(sum(item["normalized_score"] for item in datasets) / len(datasets)),
        "raw_loss": float(sum(item["loss"] for item in datasets) / len(datasets)),
        "expert_raw_loss": float(sum(item["expert_loss"] for item in datasets) / len(datasets)),
        "datasets": datasets,
    }


def visible_specs(input_dir: Path, answer_key: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": dataset_id,
            "path": str(input_dir / "datasets" / f"{dataset_id}.csv"),
            "dim": int(answer_key[dataset_id]["dim"]),
        }
        for dataset_id in VISIBLE_IDS
    ]


def hidden_specs(reference_dir: Path) -> list[dict[str, Any]]:
    manifest = load_json(reference_dir / "hidden_input" / "manifest.json")
    specs = []
    for item in manifest["datasets"]:
        path = Path(str(item["path"]))
        specs.append(
            {
                "id": str(item["id"]),
                "path": str(reference_dir / "hidden_input" / "datasets" / path.name),
                "dim": int(item["dim"]),
            }
        )
    return specs


def run_estimator(
    estimator_path: Path,
    dataset_specs: list[dict[str, Any]],
    *,
    harness_path: Path,
    timeout_sec: int,
) -> dict[str, Any]:
    if not estimator_path.is_file():
        raise FileNotFoundError(f"estimator.R missing: {estimator_path}")
    with tempfile.TemporaryDirectory(prefix="agenthle_ddm_estimator_") as temp_root:
        temp_dir = Path(temp_root)
        specs_path = temp_dir / "dataset_specs.tsv"
        output_path = temp_dir / "answers.json"
        with specs_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "path", "dim"], delimiter="\t")
            writer.writeheader()
            for spec in dataset_specs:
                writer.writerow(spec)
        proc = subprocess.run(
            ["Rscript", str(harness_path), str(estimator_path), str(specs_path), str(output_path)],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        if proc.returncode != 0:
            message = proc.stderr.strip() or proc.stdout.strip() or "unknown R error"
            raise RuntimeError(f"estimator failed with exit code {proc.returncode}: {message}")
        payload = load_json(output_path)
    return as_answers(payload)


def reference_report_path(reference_dir: Path, dataset_set: str) -> Path:
    return reference_dir / f"expert_{dataset_set}_report.json"


def prepare_reference_reports(args: argparse.Namespace) -> dict[str, Any]:
    reference_dir = Path(args.reference_dir)
    input_dir = Path(args.input_dir)
    expert = reference_dir / "expert_estimator.R"
    public_key = load_json(reference_dir / "answer_key.json")["datasets"]
    hidden_key = load_json(reference_dir / "hidden_answer_key.json")["datasets"]
    harness = reference_dir / "run_estimator_function.R"

    public_answers = run_estimator(
        expert,
        visible_specs(input_dir, public_key),
        harness_path=harness,
        timeout_sec=args.timeout_sec,
    )
    hidden_dataset_specs = hidden_specs(reference_dir)
    hidden_answers = run_estimator(
        expert,
        hidden_dataset_specs,
        harness_path=harness,
        timeout_sec=args.timeout_sec,
    )
    public_report = score_answers(public_answers, answer_key=public_key, dataset_ids=list(VISIBLE_IDS))
    hidden_report = score_answers(
        hidden_answers,
        answer_key=hidden_key,
        dataset_ids=[str(item["id"]) for item in hidden_dataset_specs],
    )
    write_json(reference_report_path(reference_dir, "public"), public_report)
    write_json(reference_report_path(reference_dir, "hidden"), hidden_report)
    return {"expert_public": public_report, "expert_hidden": hidden_report}


def generate_answer(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = Path(args.input_dir)
    reference_dir = Path(args.reference_dir)
    public_key = load_json(reference_dir / "answer_key.json")["datasets"]
    answers = run_estimator(
        Path(args.estimator),
        visible_specs(input_dir, public_key),
        harness_path=reference_dir / "run_estimator_function.R",
        timeout_sec=args.timeout_sec,
    )
    payload = {"answers": answers}
    write_json(Path(args.output), payload)
    return payload


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    reference_dir = Path(args.reference_dir)
    answer_path = output_dir / "answer.json"
    estimator_path = output_dir / "estimator.R"
    if not answer_path.is_file():
        raise FileNotFoundError(f"answer.json missing: {answer_path}")
    if not estimator_path.is_file():
        raise FileNotFoundError(f"estimator.R missing: {estimator_path}")

    public_key = load_json(reference_dir / "answer_key.json")["datasets"]
    hidden_key = load_json(reference_dir / "hidden_answer_key.json")["datasets"]
    harness = reference_dir / "run_estimator_function.R"

    submitted = as_answers(load_json(answer_path))
    generated_visible = run_estimator(
        estimator_path,
        visible_specs(input_dir, public_key),
        harness_path=harness,
        timeout_sec=args.timeout_sec,
    )

    consistency_error = ""
    consistency_score = 1.0
    for dataset_id in VISIBLE_IDS:
        dim = int(public_key[dataset_id]["dim"])
        left = as_vector(submitted.get(dataset_id), dataset_id=dataset_id, dim=dim)
        right = as_vector(generated_visible.get(dataset_id), dataset_id=dataset_id, dim=dim)
        if not allclose(left, right):
            consistency_score = 0.0
            consistency_error = f"{dataset_id}: answer.json does not match estimator.R output"
            break

    candidate_public = score_answers(submitted, answer_key=public_key, dataset_ids=list(VISIBLE_IDS))
    hidden_dataset_specs = hidden_specs(reference_dir)
    generated_hidden = run_estimator(
        estimator_path,
        hidden_dataset_specs,
        harness_path=harness,
        timeout_sec=args.timeout_sec,
    )
    candidate_hidden = score_answers(
        generated_hidden,
        answer_key=hidden_key,
        dataset_ids=[str(item["id"]) for item in hidden_dataset_specs],
    )

    public_report_file = reference_report_path(reference_dir, "public")
    hidden_report_file = reference_report_path(reference_dir, "hidden")
    if public_report_file.is_file() and hidden_report_file.is_file():
        expert_public = load_json(public_report_file)
        expert_hidden = load_json(hidden_report_file)
    else:
        prepared = prepare_reference_reports(args)
        expert_public = prepared["expert_public"]
        expert_hidden = prepared["expert_hidden"]

    public_summary = summarize_against_expert(
        candidate_public,
        expert_public,
        source="public_answer_json",
    )
    hidden_summary = summarize_against_expert(
        candidate_hidden,
        expert_hidden,
        source="hidden_estimator_runtime",
    )
    final_score = float((consistency_score + public_summary["score"] + hidden_summary["score"]) / 3.0)
    return {
        "score_definition": (
            "final_score = mean([consistency_score, public_answer.score, "
            "hidden_estimator.score]); section scores use mean min(1, expert_loss / max(candidate_loss, EPSILON))"
        ),
        "final_score": final_score,
        "score": final_score,
        "pass_threshold": 0.95,
        "consistency_score": consistency_score,
        "public_consistency": {"passed": consistency_score == 1.0, "error": consistency_error},
        "score_components": {
            "consistency_score": consistency_score,
            "public_answer_score": public_summary["score"],
            "hidden_estimator_score": hidden_summary["score"],
        },
        "public_answer": public_summary,
        "hidden_estimator": hidden_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--input-dir", required=True)
        subparser.add_argument("--reference-dir", required=True)
        subparser.add_argument("--timeout-sec", type=int, default=DEFAULT_TIMEOUT_SEC)

    prep = subparsers.add_parser("prepare-reference")
    add_common(prep)
    prep.set_defaults(func=prepare_reference_reports)

    gen = subparsers.add_parser("generate-answer")
    add_common(gen)
    gen.add_argument("--estimator", required=True)
    gen.add_argument("--output", required=True)
    gen.set_defaults(func=generate_answer)

    eval_parser = subparsers.add_parser("evaluate")
    add_common(eval_parser)
    eval_parser.add_argument("--output-dir", required=True)
    eval_parser.set_defaults(func=evaluate)

    args = parser.parse_args()
    try:
        result = args.func(args)
        print(json.dumps(result, indent=2))
    except Exception as exc:
        print(json.dumps({"score": 0.0, "final_score": 0.0, "error": str(exc)}, indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
