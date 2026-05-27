#!/usr/bin/env python
from __future__ import annotations

"""Validate STARsolo outputs for the pbmc_1k_v3 task."""

import argparse
import csv
import gzip
import json
from pathlib import Path


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _pick_existing(base: Path, names: list[str]) -> Path | None:
    for name in names:
        candidate = base / name
        if candidate.exists():
            return candidate
    return None


def _parse_number(value: str) -> float:
    cleaned = value.strip().replace(",", "")
    if cleaned.endswith("%"):
        cleaned = cleaned[:-1]
    return float(cleaned)


def _load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_spec(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_summary(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 2:
                continue
            key = row[0].strip()
            value = row[1].strip()
            if not key or key.lower() == "metric":
                continue
            try:
                metrics[key] = _parse_number(value)
            except ValueError:
                continue
    return metrics


def _metric(metrics: dict[str, float], *aliases: str) -> float | None:
    for alias in aliases:
        if alias in metrics:
            return metrics[alias]
    return None


def _metric_contains(metrics: dict[str, float], *needles: str) -> float | None:
    lowered_needles = [needle.lower() for needle in needles]
    for key, value in metrics.items():
        lowered_key = key.lower()
        if all(needle in lowered_key for needle in lowered_needles):
            return value
    return None


def _parse_log_metrics(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or "|" not in line:
                continue
            key, value = [part.strip() for part in line.split("|", 1)]
            try:
                metrics[key] = _parse_number(value)
            except ValueError:
                continue
    return metrics


def _parse_matrix_header(path: Path) -> tuple[int, int, int]:
    with _open_text(path) as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue
            rows, cols, nnz = stripped.split()
            return int(rows), int(cols), int(nnz)
    raise ValueError(f"could not find Matrix Market dimensions in {path}")


def _count_lines(path: Path) -> int:
    with _open_text(path) as handle:
        return sum(1 for _ in handle)


def _find_matrix_dir(output_dir: Path) -> Path | None:
    candidates = [
        output_dir / "filtered_feature_bc_matrix",
        output_dir / "Solo.out" / "Gene" / "filtered",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _find_summary_csv(output_dir: Path) -> Path | None:
    candidates = sorted(path for path in output_dir.rglob("*.csv") if path.is_file())
    if not candidates:
        return None

    for candidate in candidates:
        if "summary" in candidate.name.lower():
            return candidate
    return candidates[0]


def _reads_mapped_to_genome_percent(log_metrics: dict[str, float], summary_metrics: dict[str, float]) -> float | None:
    summary_direct = _metric(summary_metrics, "Reads Mapped to Genome", "Reads mapped to genome")
    if summary_direct is None:
        summary_direct = _metric_contains(summary_metrics, "Reads", "Mapped", "Genome")
    if summary_direct is not None:
        return summary_direct

    total = 0.0
    found = False
    for key in (
        "Uniquely mapped reads %",
        "% of reads mapped to multiple loci",
        "% of reads mapped to too many loci",
    ):
        value = log_metrics.get(key)
        if value is not None:
            total += value
            found = True
    if found:
        return total
    return None


def _fraction_reads_in_cells_percent(summary_metrics: dict[str, float]) -> float | None:
    direct = _metric(
        summary_metrics,
        "Fraction of Unique Reads in Cells",
        "Fraction of Reads in Cells",
        "Fraction Reads in Cells",
    )
    if direct is not None:
        return direct
    return _metric_contains(summary_metrics, "Fraction", "Reads", "Cells")


def evaluate(output_dir: Path, spec_path: Path) -> dict:
    reasons: list[str] = []
    details: dict[str, object] = {}
    spec = _load_spec(spec_path)
    required_manifest = spec["required_manifest"]
    thresholds = spec["thresholds"]

    manifest_path = output_dir / "run_manifest.json"
    summary_path = _find_summary_csv(output_dir)
    log_path = output_dir / "Log.final.out"

    matrix_dir = _find_matrix_dir(output_dir)
    if matrix_dir is None:
        reasons.append("missing filtered_feature_bc_matrix directory")
        return {"score": 0.0, "passed": False, "reasons": reasons, "details": details}

    matrix_path = _pick_existing(matrix_dir, ["matrix.mtx", "matrix.mtx.gz"])
    barcodes_path = _pick_existing(matrix_dir, ["barcodes.tsv", "barcodes.tsv.gz"])
    features_path = _pick_existing(matrix_dir, ["features.tsv", "features.tsv.gz"])

    required_files = {
        "matrix": matrix_path,
        "barcodes": barcodes_path,
        "features": features_path,
        "summary_csv": summary_path if summary_path is not None and summary_path.exists() else None,
        "log_final_out": log_path if log_path.exists() else None,
        "run_manifest": manifest_path if manifest_path.exists() else None,
    }
    missing = [name for name, path in required_files.items() if path is None]
    if missing:
        reasons.append(f"missing required files: {', '.join(missing)}")
        return {"score": 0.0, "passed": False, "reasons": reasons, "details": details}

    rows, cols, nnz = _parse_matrix_header(matrix_path)
    barcode_count = _count_lines(barcodes_path)
    feature_count = _count_lines(features_path)
    details.update(
        {
            "matrix_rows": rows,
            "matrix_cols": cols,
            "matrix_nnz": nnz,
            "barcode_count": barcode_count,
            "feature_count": feature_count,
            "matrix_dir": str(matrix_dir),
            "summary_csv": str(summary_path),
        }
    )

    if rows <= 0 or cols <= 0:
        reasons.append("matrix dimensions must be non-zero")
    if nnz <= 0:
        reasons.append("matrix must contain non-zero entries")
    if cols < thresholds["estimated_cells_min"] or cols > thresholds["estimated_cells_max"]:
        reasons.append(
            f"cell count {cols} outside {thresholds['estimated_cells_min']}-{thresholds['estimated_cells_max']}"
        )
    if rows < thresholds["detected_genes_min"]:
        reasons.append(f"detected genes {rows} below {thresholds['detected_genes_min']}")
    if barcode_count != cols:
        reasons.append(f"barcode count {barcode_count} does not match matrix columns {cols}")
    if feature_count != rows:
        reasons.append(f"feature count {feature_count} does not match matrix rows {rows}")

    summary_metrics = _load_summary(summary_path)
    log_metrics = _parse_log_metrics(log_path)
    details["summary_metrics"] = summary_metrics

    estimated_cells = _metric(summary_metrics, "Estimated Number of Cells")
    if estimated_cells is None:
        estimated_cells = _metric_contains(summary_metrics, "Estimated", "Cells")

    median_genes = _metric(summary_metrics, "Median Gene per Cell", "Median Genes per Cell")
    if median_genes is None:
        median_genes = _metric_contains(summary_metrics, "Median", "Gene")

    median_umi = _metric(summary_metrics, "Median UMI per Cell", "Median UMI Counts per Cell")
    if median_umi is None:
        median_umi = _metric_contains(summary_metrics, "Median", "UMI")

    reads_mapped = _reads_mapped_to_genome_percent(log_metrics, summary_metrics)
    fraction_reads_in_cells = _fraction_reads_in_cells_percent(summary_metrics)

    if estimated_cells is None:
        reasons.append("missing Estimated Number of Cells metric")
    elif not (thresholds["estimated_cells_min"] <= estimated_cells <= thresholds["estimated_cells_max"]):
        reasons.append(
            "Estimated Number of Cells "
            f"{estimated_cells} outside {thresholds['estimated_cells_min']}-{thresholds['estimated_cells_max']}"
        )

    if median_genes is None:
        reasons.append("missing Median Gene per Cell metric")
    elif not (
        thresholds["median_genes_per_cell_min"]
        <= median_genes
        <= thresholds["median_genes_per_cell_max"]
    ):
        reasons.append(
            "Median Gene per Cell "
            f"{median_genes} outside "
            f"{thresholds['median_genes_per_cell_min']}-{thresholds['median_genes_per_cell_max']}"
        )

    if median_umi is None:
        reasons.append("missing Median UMI per Cell metric")
    elif not (
        thresholds["median_umi_per_cell_min"]
        <= median_umi
        <= thresholds["median_umi_per_cell_max"]
    ):
        reasons.append(
            "Median UMI per Cell "
            f"{median_umi} outside "
            f"{thresholds['median_umi_per_cell_min']}-{thresholds['median_umi_per_cell_max']}"
        )

    if reads_mapped is None:
        reasons.append("missing reads-mapped-to-genome metric")
    elif reads_mapped < thresholds["reads_mapped_to_genome_min_percent"]:
        reasons.append(
            "reads mapped to genome "
            f"{reads_mapped}% below {thresholds['reads_mapped_to_genome_min_percent']}%"
        )

    if fraction_reads_in_cells is None:
        reasons.append("missing fraction-reads-in-cells metric")
    elif fraction_reads_in_cells < thresholds["fraction_reads_in_cells_min_percent"]:
        reasons.append(
            "fraction reads in cells "
            f"{fraction_reads_in_cells}% below {thresholds['fraction_reads_in_cells_min_percent']}%"
        )

    manifest = _load_manifest(manifest_path)
    details["manifest"] = manifest
    for key, expected_value in required_manifest.items():
        actual_value = manifest.get(key)
        if actual_value != expected_value:
            reasons.append(f"manifest field {key!r} expected {expected_value!r}, found {actual_value!r}")

    passed = not reasons
    return {
        "score": 1.0 if passed else 0.0,
        "passed": passed,
        "reasons": reasons,
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--spec", required=True)
    args = parser.parse_args()

    result = evaluate(Path(args.output_dir), Path(args.spec))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
