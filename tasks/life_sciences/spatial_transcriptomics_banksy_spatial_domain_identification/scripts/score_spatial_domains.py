"""Score BANKSY spatial-domain identification output bundles."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

SLICE_CONFIG: tuple[tuple[str, int], ...] = (
    ("151507", 7),
    ("151508", 7),
    ("151509", 7),
    ("151510", 7),
    ("151673", 7),
    ("151674", 7),
    ("151675", 7),
    ("151676", 7),
    ("151669", 5),
    ("151670", 5),
    ("151671", 5),
    ("151672", 5),
)

SUMMARY_HEADER = ["slice_id", "n_spots_scored", "n_clusters_pred", "ari", "nmi"]
LABEL_HEADER = ["barcode", "predicted_label"]
REQUIRED_PNG = "umap_overlay_151673.png"
FULL_CREDIT_MEDIAN_ARI = 0.45
FULL_CREDIT_MIN_SLICE_ARI = 0.15
PARTIAL_CREDIT_MEDIAN_ARI = 0.30


@dataclass(frozen=True)
class SliceScore:
    slice_id: str
    n_barcodes: int
    n_reference_scored: int
    n_clusters_pred: int
    ari: float
    nmi: float


@dataclass(frozen=True)
class ScoreResult:
    score: float
    passed: bool
    reason: str
    median_ari: float = 0.0
    median_nmi: float = 0.0
    min_ari: float = 0.0
    slice_scores: tuple[SliceScore, ...] = ()

    def to_dict(self) -> dict:
        data = asdict(self)
        data["slice_scores"] = [asdict(item) for item in self.slice_scores]
        return data


def _parse_csv(text: str, *, expected_header: list[str], label: str, delimiter: str = ",") -> list[dict[str, str]]:
    reader = csv.DictReader(text.splitlines(), delimiter=delimiter)
    if reader.fieldnames != expected_header:
        raise ValueError(f"{label} has header {reader.fieldnames!r}; expected {expected_header!r}")
    return list(reader)


def _parse_float(value: str, *, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric: {value!r}") from exc


def _parse_int(value: str, *, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not an integer: {value!r}") from exc


def _read_barcodes(text: str, *, label: str) -> list[str]:
    barcodes = [line.strip() for line in text.splitlines() if line.strip()]
    if not barcodes:
        raise ValueError(f"{label} is empty")
    if len(barcodes) != len(set(barcodes)):
        raise ValueError(f"{label} contains duplicate barcodes")
    return barcodes


def _validate_manifest(manifest_text: str) -> None:
    data = json.loads(manifest_text)
    if not isinstance(data, dict):
        raise ValueError("manifest.json must contain a JSON object")
    for key in ["has_graph", "has_embedding", "has_clustering"]:
        if data.get(key) is not True:
            raise ValueError(f"manifest.json field {key!r} must be true")
    if not isinstance(data.get("seed"), int):
        raise ValueError("manifest.json field 'seed' must be an integer")
    if not isinstance(data.get("method"), str) or not data["method"].strip():
        raise ValueError("manifest.json field 'method' must be a nonempty string")
    if not isinstance(data.get("lambda"), (int, float)):
        raise ValueError("manifest.json field 'lambda' must be numeric")
    if not isinstance(data.get("k_geom"), int):
        raise ValueError("manifest.json field 'k_geom' must be an integer")
    if not isinstance(data.get("hvg_flavor"), str) or not data["hvg_flavor"].strip():
        raise ValueError("manifest.json field 'hvg_flavor' must be a nonempty string")
    if not isinstance(data.get("clustering_backend"), str) or not data["clustering_backend"].strip():
        raise ValueError("manifest.json field 'clustering_backend' must be a nonempty string")
    software_versions = data.get("software_versions")
    if not isinstance(software_versions, dict) or not software_versions:
        raise ValueError("manifest.json field 'software_versions' must be a nonempty object")


def _validate_png(png_bytes: bytes) -> None:
    if len(png_bytes) <= 50_000:
        raise ValueError(f"{REQUIRED_PNG} is too small: {len(png_bytes)} bytes")
    if not png_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"{REQUIRED_PNG} is not a valid PNG")


def _validate_run_sh(text: str) -> None:
    if not text.strip():
        raise ValueError("run.sh must be non-empty")


def _score_from_metrics(median_ari: float, min_ari: float) -> tuple[float, str]:
    if median_ari >= FULL_CREDIT_MEDIAN_ARI and min_ari >= FULL_CREDIT_MIN_SLICE_ARI:
        return 1.0, "passed"
    if median_ari >= PARTIAL_CREDIT_MEDIAN_ARI:
        return 0.5, "partial_median_ari"
    return 0.0, "median_ari_below_partial_threshold"


def score_output_bundle(
    *,
    summary_csv: str,
    manifest_json: str,
    per_slice_labels: Mapping[str, str],
    reference_annotations: Mapping[str, str],
    input_barcodes: Mapping[str, str],
    umap_png: bytes,
    run_sh: str,
) -> ScoreResult:
    """Score one candidate output bundle against hidden per-slice annotations."""

    try:
        _validate_manifest(manifest_json)
        _validate_png(umap_png)
        _validate_run_sh(run_sh)

        summary_rows = _parse_csv(summary_csv, expected_header=SUMMARY_HEADER, label="summary.csv")
        if len(summary_rows) != len(SLICE_CONFIG):
            raise ValueError(f"summary.csv must contain {len(SLICE_CONFIG)} rows, found {len(summary_rows)}")
        summary_by_slice = {row["slice_id"]: row for row in summary_rows}
        expected_slice_ids = {slice_id for slice_id, _ in SLICE_CONFIG}
        if set(summary_by_slice) != expected_slice_ids:
            raise ValueError(
                f"summary.csv slice IDs mismatch: expected {sorted(expected_slice_ids)}, "
                f"found {sorted(summary_by_slice)}"
            )

        slice_scores: list[SliceScore] = []
        for slice_id, expected_clusters in SLICE_CONFIG:
            expected_barcodes = _read_barcodes(input_barcodes[slice_id], label=f"input/data/{slice_id}/barcodes.tsv")
            label_rows = _parse_csv(
                per_slice_labels[slice_id],
                expected_header=LABEL_HEADER,
                label=f"per_slice/{slice_id}_labels.csv",
            )
            truth_rows = _parse_csv(
                reference_annotations[slice_id],
                expected_header=["barcode", "layer"],
                label=f"reference/manual_annotations/{slice_id}_manual_annotations.tsv",
                delimiter="\t",
            )

            predictions = {row["barcode"]: row["predicted_label"] for row in label_rows}
            duplicate_count = len(label_rows) - len(predictions)
            if duplicate_count:
                raise ValueError(f"{slice_id} labels contain {duplicate_count} duplicate barcodes")
            if set(predictions) != set(expected_barcodes):
                missing = sorted(set(expected_barcodes) - set(predictions))[:5]
                extra = sorted(set(predictions) - set(expected_barcodes))[:5]
                raise ValueError(f"{slice_id} labels must match staged barcodes exactly; missing={missing} extra={extra}")

            parsed_labels: dict[str, int] = {}
            for barcode, label in predictions.items():
                parsed = _parse_int(label, label=f"{slice_id} predicted_label for {barcode}")
                if parsed < 1 or parsed > expected_clusters:
                    raise ValueError(f"{slice_id} predicted_label {parsed} outside 1..{expected_clusters}")
                parsed_labels[barcode] = parsed
            n_clusters = len(set(parsed_labels.values()))
            if n_clusters != expected_clusters:
                raise ValueError(f"{slice_id} predicted {n_clusters} clusters; expected {expected_clusters}")

            truth = {row["barcode"]: row["layer"] for row in truth_rows}
            if set(truth) != set(expected_barcodes):
                raise ValueError(f"{slice_id} hidden annotations do not match staged barcodes")
            scored_barcodes = [barcode for barcode in expected_barcodes if truth[barcode] != "NA"]
            if not scored_barcodes:
                raise ValueError(f"{slice_id} has no scored reference spots")
            truth_labels = [truth[barcode] for barcode in scored_barcodes]
            pred_labels = [str(parsed_labels[barcode]) for barcode in scored_barcodes]
            ari = float(adjusted_rand_score(truth_labels, pred_labels))
            nmi = float(normalized_mutual_info_score(truth_labels, pred_labels))

            summary = summary_by_slice[slice_id]
            if _parse_int(summary["n_spots_scored"], label=f"{slice_id} n_spots_scored") != len(expected_barcodes):
                raise ValueError(f"{slice_id} summary n_spots_scored must equal staged barcode count")
            if _parse_int(summary["n_clusters_pred"], label=f"{slice_id} n_clusters_pred") != expected_clusters:
                raise ValueError(f"{slice_id} summary n_clusters_pred must equal target cluster count")
            _parse_float(summary["ari"], label=f"{slice_id} ari")
            _parse_float(summary["nmi"], label=f"{slice_id} nmi")

            slice_scores.append(
                SliceScore(
                    slice_id=slice_id,
                    n_barcodes=len(expected_barcodes),
                    n_reference_scored=len(scored_barcodes),
                    n_clusters_pred=n_clusters,
                    ari=ari,
                    nmi=nmi,
                )
            )

        median_ari = statistics.median(item.ari for item in slice_scores)
        median_nmi = statistics.median(item.nmi for item in slice_scores)
        min_ari = min(item.ari for item in slice_scores)
        score, reason = _score_from_metrics(median_ari, min_ari)
        return ScoreResult(
            score=score,
            passed=score > 0.0,
            reason=reason,
            median_ari=median_ari,
            median_nmi=median_nmi,
            min_ari=min_ari,
            slice_scores=tuple(slice_scores),
        )
    except Exception as exc:
        return ScoreResult(score=0.0, passed=False, reason=str(exc))


def score_output_dir(output_dir: Path, reference_dir: Path, input_dir: Path) -> ScoreResult:
    per_slice = {
        slice_id: (output_dir / "per_slice" / f"{slice_id}_labels.csv").read_text(encoding="utf-8")
        for slice_id, _ in SLICE_CONFIG
    }
    annotations = {
        slice_id: (reference_dir / "manual_annotations" / f"{slice_id}_manual_annotations.tsv").read_text(
            encoding="utf-8"
        )
        for slice_id, _ in SLICE_CONFIG
    }
    barcodes = {
        slice_id: (input_dir / "data" / slice_id / "barcodes.tsv").read_text(encoding="utf-8")
        for slice_id, _ in SLICE_CONFIG
    }
    return score_output_bundle(
        summary_csv=(output_dir / "summary.csv").read_text(encoding="utf-8"),
        manifest_json=(output_dir / "manifest.json").read_text(encoding="utf-8"),
        per_slice_labels=per_slice,
        reference_annotations=annotations,
        input_barcodes=barcodes,
        umap_png=(output_dir / REQUIRED_PNG).read_bytes(),
        run_sh=(output_dir / "run.sh").read_text(encoding="utf-8"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    args = parser.parse_args()
    result = score_output_dir(args.output_dir, args.reference_dir, args.input_dir)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
