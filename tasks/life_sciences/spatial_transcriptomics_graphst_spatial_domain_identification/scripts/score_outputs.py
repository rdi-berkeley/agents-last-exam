"""Local evaluator for the GraphST spatial-domain identification task."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

EXPECTED_SLICE_IDS = (
    "151507",
    "151508",
    "151509",
    "151510",
    "151669",
    "151670",
    "151671",
    "151672",
    "151673",
    "151674",
    "151675",
    "151676",
)
EXPECTED_SUMMARY_HEADER = [
    "slice_id",
    "n_spots_scored",
    "n_clusters_pred",
    "ari",
    "nmi",
]
EXPECTED_LABEL_HEADER = ["barcode", "predicted_label"]
PNG_MAGIC_HEX = "89504e470d0a1a0a"
MIN_PNG_BYTES = 50_000
MIN_LABEL_ROWS = 3_000
FULL_CREDIT_MEDIAN_ARI = 0.55
FULL_CREDIT_MIN_SLICE_ARI = 0.20
PARTIAL_CREDIT_MEDIAN_ARI = 0.40


@dataclass
class ScoreResult:
    score: float
    passed: bool
    reason: str
    hard_gate: str | None
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _hard_fail(reason: str, details: dict[str, Any] | None = None) -> ScoreResult:
    return ScoreResult(0.0, False, reason, reason, details or {})


def _comb2(value: int) -> int:
    return value * (value - 1) // 2 if value >= 2 else 0


def _parse_csv_rows(
    text: str,
    *,
    delimiter: str = ",",
) -> tuple[list[str], list[dict[str, str]]]:
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")), delimiter=delimiter)
    if reader.fieldnames is None:
        raise ValueError("missing header row")
    header = [field.strip() for field in reader.fieldnames]
    rows = [{key.strip(): value for key, value in row.items()} for row in reader]
    return header, rows


def _parse_slice_config(text: str) -> dict[str, int]:
    header, rows = _parse_csv_rows(text)
    expected = ["slice_id", "subject_id", "n_clusters"]
    if header != expected:
        raise ValueError(f"slice_config.csv header must be {expected}, got {header}")
    config: dict[str, int] = {}
    for row in rows:
        slice_id = row["slice_id"].strip()
        if slice_id in config:
            raise ValueError(f"duplicate slice_id in slice_config.csv: {slice_id}")
        config[slice_id] = int(row["n_clusters"])
    if tuple(sorted(config)) != EXPECTED_SLICE_IDS:
        raise ValueError("slice_config.csv does not match the expected 12 slice ids")
    return config


def _parse_barcodes(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _parse_manifest(text: str) -> dict[str, Any]:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("manifest.json must contain a JSON object")
    required_true = ("has_graph", "has_embedding", "has_clustering")
    for key in required_true:
        if payload.get(key) is not True:
            raise ValueError(f"manifest.json field {key!r} must be true")
    if not isinstance(payload.get("seed"), int):
        raise ValueError("manifest.json field 'seed' must be an integer")
    if payload.get("epochs") != 600:
        raise ValueError("manifest.json field 'epochs' must equal 600")
    for key in (
        "method",
        "graphst_commit",
        "torch_version",
        "device",
        "hvg_flavor",
        "clustering_backend",
    ):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"manifest.json field {key!r} must be a non-empty string")
    return payload


def _parse_summary(text: str) -> dict[str, dict[str, str]]:
    header, rows = _parse_csv_rows(text)
    if header != EXPECTED_SUMMARY_HEADER:
        raise ValueError(f"summary.csv header must be {EXPECTED_SUMMARY_HEADER}, got {header}")
    if len(rows) != len(EXPECTED_SLICE_IDS):
        raise ValueError(f"summary.csv must contain {len(EXPECTED_SLICE_IDS)} rows")
    summary: dict[str, dict[str, str]] = {}
    for row in rows:
        slice_id = row["slice_id"].strip()
        if slice_id in summary:
            raise ValueError(f"duplicate summary row for slice_id={slice_id}")
        summary[slice_id] = row
        int(row["n_spots_scored"])
        int(row["n_clusters_pred"])
        float(row["ari"])
        float(row["nmi"])
    if tuple(sorted(summary)) != EXPECTED_SLICE_IDS:
        raise ValueError("summary.csv does not match the expected 12 slice ids")
    return summary


def _parse_label_file(text: str, *, expected_clusters: int) -> tuple[dict[str, int], list[int]]:
    header, rows = _parse_csv_rows(text)
    if header != EXPECTED_LABEL_HEADER:
        raise ValueError(f"labels header must be {EXPECTED_LABEL_HEADER}, got {header}")
    if len(rows) < MIN_LABEL_ROWS:
        raise ValueError(f"labels file must contain at least {MIN_LABEL_ROWS} rows")

    labels: dict[str, int] = {}
    observed_values: list[int] = []
    for row in rows:
        barcode = row["barcode"].strip()
        if not barcode:
            raise ValueError("labels file contains an empty barcode")
        if barcode in labels:
            raise ValueError(f"duplicate barcode in labels file: {barcode}")
        try:
            predicted_label = int(row["predicted_label"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"predicted_label must be an integer for barcode {barcode}") from exc
        if predicted_label < 1 or predicted_label > expected_clusters:
            raise ValueError(f"predicted_label {predicted_label} is outside 1..{expected_clusters}")
        labels[barcode] = predicted_label
        observed_values.append(predicted_label)

    if len(set(observed_values)) != expected_clusters:
        raise ValueError(
            f"labels file must contain exactly {expected_clusters} unique predicted labels"
        )
    return labels, observed_values


def _parse_reference_annotations(text: str) -> list[tuple[str, str]]:
    header, rows = _parse_csv_rows(text, delimiter="\t")
    if header != ["barcode", "layer"]:
        raise ValueError(f"manual annotations header must be ['barcode', 'layer'], got {header}")
    annotations: list[tuple[str, str]] = []
    for row in rows:
        barcode = row["barcode"].strip()
        layer = row["layer"].strip()
        if barcode:
            annotations.append((barcode, layer))
    return annotations


def _adjusted_rand_index(y_true: list[str], y_pred: list[str]) -> float:
    n_samples = len(y_true)
    if n_samples != len(y_pred):
        raise ValueError("ARI inputs must have equal length")
    if n_samples < 2:
        return 1.0

    contingency: dict[str, Counter[str]] = defaultdict(Counter)
    true_counts: Counter[str] = Counter()
    pred_counts: Counter[str] = Counter()
    for truth, pred in zip(y_true, y_pred):
        contingency[truth][pred] += 1
        true_counts[truth] += 1
        pred_counts[pred] += 1

    sum_comb = sum(_comb2(count) for bucket in contingency.values() for count in bucket.values())
    sum_true = sum(_comb2(count) for count in true_counts.values())
    sum_pred = sum(_comb2(count) for count in pred_counts.values())
    total_pairs = _comb2(n_samples)
    if total_pairs == 0:
        return 1.0

    expected_index = (sum_true * sum_pred) / total_pairs
    max_index = 0.5 * (sum_true + sum_pred)
    denominator = max_index - expected_index
    if denominator == 0:
        return 1.0
    return (sum_comb - expected_index) / denominator


def _normalized_mutual_info(y_true: list[str], y_pred: list[str]) -> float:
    n_samples = len(y_true)
    if n_samples != len(y_pred):
        raise ValueError("NMI inputs must have equal length")
    if n_samples == 0:
        return 0.0
    if len(set(y_true)) == 1 and len(set(y_pred)) == 1:
        return 1.0

    contingency: dict[str, Counter[str]] = defaultdict(Counter)
    true_counts: Counter[str] = Counter(y_true)
    pred_counts: Counter[str] = Counter(y_pred)
    for truth, pred in zip(y_true, y_pred):
        contingency[truth][pred] += 1

    n_float = float(n_samples)
    mutual_info = 0.0
    for truth, pred_counts_for_truth in contingency.items():
        for pred, count in pred_counts_for_truth.items():
            if count == 0:
                continue
            pij = count / n_float
            pi = true_counts[truth] / n_float
            pj = pred_counts[pred] / n_float
            mutual_info += pij * math.log(pij / (pi * pj))

    def _entropy(counts: Counter[str]) -> float:
        entropy = 0.0
        for count in counts.values():
            if count == 0:
                continue
            probability = count / n_float
            entropy -= probability * math.log(probability)
        return entropy

    h_true = _entropy(true_counts)
    h_pred = _entropy(pred_counts)
    if h_true == 0.0 and h_pred == 0.0:
        return 1.0
    denominator = h_true + h_pred
    if denominator == 0.0:
        return 0.0
    return 2.0 * mutual_info / denominator


def score_output_bundle(
    *,
    slice_config_csv: str,
    summary_csv: str,
    manifest_json: str,
    label_csvs: dict[str, str],
    barcode_tsvs: dict[str, str],
    reference_annotation_tsvs: dict[str, str],
    overlay_probe: dict[str, Any],
) -> ScoreResult:
    try:
        slice_config = _parse_slice_config(slice_config_csv)
        summary_rows = _parse_summary(summary_csv)
        manifest = _parse_manifest(manifest_json)
    except (ValueError, json.JSONDecodeError) as exc:
        return _hard_fail(str(exc))

    size_bytes = overlay_probe.get("size_bytes")
    magic_hex = overlay_probe.get("magic_hex")
    if not isinstance(size_bytes, int) or size_bytes < MIN_PNG_BYTES:
        return _hard_fail(
            "umap_overlay_too_small",
            {"size_bytes": size_bytes, "minimum_size_bytes": MIN_PNG_BYTES},
        )
    if magic_hex != PNG_MAGIC_HEX:
        return _hard_fail("umap_overlay_not_png", {"magic_hex": magic_hex})

    slice_metrics: dict[str, dict[str, float | int]] = {}
    aris: list[float] = []
    nmis: list[float] = []

    for slice_id in EXPECTED_SLICE_IDS:
        if slice_id not in label_csvs:
            return _hard_fail("missing_label_file", {"slice_id": slice_id})
        if slice_id not in barcode_tsvs:
            return _hard_fail("missing_barcode_file", {"slice_id": slice_id})
        if slice_id not in reference_annotation_tsvs:
            return _hard_fail("missing_reference_annotations", {"slice_id": slice_id})

        expected_clusters = slice_config[slice_id]
        try:
            predicted_labels, observed_values = _parse_label_file(
                label_csvs[slice_id],
                expected_clusters=expected_clusters,
            )
        except ValueError as exc:
            return _hard_fail(f"invalid_label_file_{slice_id}", {"error": str(exc)})

        input_barcodes = _parse_barcodes(barcode_tsvs[slice_id])
        input_barcode_set = set(input_barcodes)
        if len(input_barcode_set) != len(input_barcodes):
            return _hard_fail("duplicate_input_barcodes", {"slice_id": slice_id})
        if input_barcode_set != set(predicted_labels):
            return _hard_fail(
                "barcode_set_mismatch",
                {
                    "slice_id": slice_id,
                    "input_count": len(input_barcode_set),
                    "predicted_count": len(predicted_labels),
                },
            )

        summary_row = summary_rows[slice_id]
        if int(summary_row["n_spots_scored"]) != len(input_barcodes):
            return _hard_fail(
                "summary_spot_count_mismatch",
                {
                    "slice_id": slice_id,
                    "summary_n_spots_scored": int(summary_row["n_spots_scored"]),
                    "expected_n_spots_scored": len(input_barcodes),
                },
            )
        if int(summary_row["n_clusters_pred"]) != expected_clusters:
            return _hard_fail(
                "summary_cluster_count_mismatch",
                {
                    "slice_id": slice_id,
                    "summary_n_clusters_pred": int(summary_row["n_clusters_pred"]),
                    "expected_n_clusters": expected_clusters,
                },
            )

        annotations = _parse_reference_annotations(reference_annotation_tsvs[slice_id])
        y_true: list[str] = []
        y_pred: list[str] = []
        for barcode, layer in annotations:
            if layer == "NA":
                continue
            if barcode not in predicted_labels:
                return _hard_fail(
                    "reference_join_failed", {"slice_id": slice_id, "barcode": barcode}
                )
            y_true.append(layer)
            y_pred.append(str(predicted_labels[barcode]))

        if not y_true:
            return _hard_fail("no_scored_spots_after_na_filter", {"slice_id": slice_id})

        ari = _adjusted_rand_index(y_true, y_pred)
        nmi = _normalized_mutual_info(y_true, y_pred)
        aris.append(ari)
        nmis.append(nmi)
        slice_metrics[slice_id] = {
            "ari": round(ari, 6),
            "nmi": round(nmi, 6),
            "n_input_barcodes": len(input_barcodes),
            "n_scored_spots": len(y_true),
            "n_unique_predicted_labels": len(set(observed_values)),
        }

    median_ari = float(median(aris))
    median_nmi = float(median(nmis))
    passed = median_ari >= FULL_CREDIT_MEDIAN_ARI and min(aris) >= FULL_CREDIT_MIN_SLICE_ARI
    details = {
        "median_ari": round(median_ari, 6),
        "median_nmi": round(median_nmi, 6),
        "full_credit_target_met": passed,
        "partial_credit_target_met": median_ari >= PARTIAL_CREDIT_MEDIAN_ARI,
        "manifest": {
            "method": manifest["method"],
            "device": manifest["device"],
            "epochs": manifest["epochs"],
            "clustering_backend": manifest["clustering_backend"],
        },
        "overlay_probe": {
            "size_bytes": size_bytes,
            "magic_hex": magic_hex,
        },
        "slice_metrics": slice_metrics,
    }
    return ScoreResult(
        score=round(max(0.0, min(1.0, median_ari)), 6),
        passed=passed,
        reason="passed" if passed else "score_below_full_credit_target",
        hard_gate=None,
        details=details,
    )


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_overlay_probe(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"size_bytes": len(data), "magic_hex": data[:8].hex()}


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--reference-dir", required=True, type=Path)
    args = parser.parse_args()

    result = score_output_bundle(
        slice_config_csv=_read_text(args.input_dir / "data" / "slice_config.csv"),
        summary_csv=_read_text(args.output_dir / "summary.csv"),
        manifest_json=_read_text(args.output_dir / "manifest.json"),
        label_csvs={
            slice_id: _read_text(args.output_dir / "per_slice" / f"{slice_id}_labels.csv")
            for slice_id in EXPECTED_SLICE_IDS
        },
        barcode_tsvs={
            slice_id: _read_text(args.input_dir / "data" / slice_id / "barcodes.tsv")
            for slice_id in EXPECTED_SLICE_IDS
        },
        reference_annotation_tsvs={
            slice_id: _read_text(
                args.reference_dir / "manual_annotations" / f"{slice_id}_manual_annotations.tsv"
            )
            for slice_id in EXPECTED_SLICE_IDS
        },
        overlay_probe=_read_overlay_probe(args.output_dir / "umap_overlay_151673.png"),
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    _cli()
