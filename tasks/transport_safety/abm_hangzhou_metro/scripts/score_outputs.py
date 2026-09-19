"""Score the published seeded metro model, without historical answer tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

if __package__:
    from .public_model import CONTRACT_VERSION, simulate
else:
    from public_model import CONTRACT_VERSION, simulate

TRIP_KEY = ("card_id", "o_station", "d_station", "start_time")
OUTPUT_COLUMNS = [
    "card_id",
    "o_station",
    "d_station",
    "start_time",
    "end_time_simulate",
    "transfer_times",
    "end_time_real",
    "duration_simulation",
    "duration_real",
]
REPORT_KEYS = ["R2", "RMSE", "Total passengers", "std(sim-real)"]
REPORT_R2_TOL = 0.001
REPORT_METRIC_TOL = 0.02
PUBLIC_CONTRACT = Path(__file__).resolve().parents[1] / "simulation_contract.md"


@dataclass(frozen=True)
class ScoreResult:
    score: float
    passed: bool
    reason: str
    hard_gate: str | None
    details: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _failure(reason: str, hard_gate: str, **details: object) -> ScoreResult:
    return ScoreResult(0.0, False, reason, hard_gate, details)


def _parse_int_like(value: str, *, field: str) -> int:
    text = value.strip()
    if not text:
        raise ValueError(f"empty integer field: {field}")
    if "." in text or "e" in text.lower():
        parsed = float(text)
        if not math.isfinite(parsed) or not parsed.is_integer() or abs(parsed) > 2**53:
            raise ValueError(f"non-integral numeric field {field}: {value!r}")
        return int(parsed)
    return int(text)


def _parse_validation_report(path: Path) -> dict[str, float]:
    values = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in raw_line:
            continue
        key, value = [part.strip() for part in raw_line.split("=", 1)]
        if key not in REPORT_KEYS:
            continue
        if key in values:
            raise ValueError(f"duplicate report metric: {key}")
        if key == "RMSE":
            value = value.removesuffix(" minutes").strip()
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"non-finite report metric: {key}")
        if key in {"RMSE", "std(sim-real)"} and parsed < 0:
            raise ValueError(f"negative report metric: {key}")
        if key == "R2" and parsed > 1:
            raise ValueError("R2 cannot exceed one")
        if key == "Total passengers" and (not parsed.is_integer() or parsed < 0):
            raise ValueError("Total passengers must be a nonnegative integer")
        values[key] = parsed
    if set(values) != set(REPORT_KEYS):
        raise ValueError("validation report missing required metrics")
    return values


def score_output_bundle(
    *, output_dir: Path, reference_dir: Path | None = None, input_dir: Path
) -> ScoreResult:
    staged_contract = input_dir / "simulation_contract.md"
    if (
        not staged_contract.is_file()
        or staged_contract.read_bytes() != PUBLIC_CONTRACT.read_bytes()
    ):
        raise RuntimeError("evaluator/public contract mismatch: stage simulation_contract.md v2")
    for name in (
        "data/afc_hangzhou.csv",
        "gis/hangzhou_lines.json",
        "gis/hangzhou_stations.json",
        "network_config/station_sequence.csv",
        "network_config/operation_parameters.json",
    ):
        if not (input_dir / name).is_file():
            raise RuntimeError(f"evaluator-controlled input missing: {name}")
    for name in ("passenger_records.csv", "validation_report.txt", "simulation_manifest.json"):
        if not (output_dir / name).is_file():
            return _failure("missing_file", "missing_required_file", missing=name)
    try:
        manifest = json.loads((output_dir / "simulation_manifest.json").read_text())
        if not isinstance(manifest, dict) or set(manifest) != {"contract_version", "seed"}:
            raise ValueError("manifest must contain only contract_version and seed")
        if manifest["contract_version"] != CONTRACT_VERSION:
            raise ValueError("unsupported simulation contract")
        seed = manifest["seed"]
        if type(seed) is not int or not 0 <= seed < 2**64:
            raise ValueError("seed must be an unsigned 64-bit integer")
        report = _parse_validation_report(output_dir / "validation_report.txt")
    except (ValueError, TypeError, OSError) as exc:
        return _failure("metadata_parse_error", "output_contract", error=str(exc))
    try:
        simulation = simulate(input_dir, seed)
    except Exception as exc:
        raise RuntimeError(f"evaluator-controlled public model failed: {exc}") from exc
    expected = {
        trip[:4]: index
        for index, trip in enumerate(simulation.trips)
        if simulation.ends[index] is not None
    }
    if not expected or len(expected) / len(simulation.trips) < 0.8:
        raise RuntimeError("public model does not achieve the unchanged 80% coverage floor")
    count = 0
    sum_real = sum_real_sq = sum_diff = sum_diff_sq = 0
    seen = set()
    try:
        with (output_dir / "passenger_records.csv").open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != OUTPUT_COLUMNS:
                return _failure("csv_parse_error", "output_schema", observed=reader.fieldnames)
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("ragged output row")
                values = {key: _parse_int_like(row[key], field=key) for key in OUTPUT_COLUMNS[3:]}
                identity = (
                    row["card_id"],
                    row["o_station"],
                    row["d_station"],
                    values["start_time"],
                )
                if identity in seen:
                    return _failure("duplicate_trip_key", "non_unique_trip_identity")
                seen.add(identity)
                if identity not in expected:
                    return _failure("unexpected_trip", "demand_conservation", trip=list(identity))
                index = expected[identity]
                trip = simulation.trips[index]
                observed_duration = trip[4] - trip[3]
                if observed_duration < 0:
                    observed_duration += 1440
                if (
                    values["end_time_real"] != trip[4]
                    or values["duration_real"] != observed_duration
                ):
                    return _failure("visible_label_echo_mismatch", "output_contract_real_columns")
                end = simulation.ends[index]
                if (
                    values["end_time_simulate"] != end
                    or values["duration_simulation"] != end - trip[3]
                    or values["transfer_times"] != simulation.transfers[index]
                ):
                    return _failure(
                        "public_model_mismatch", "simulation_dynamics", trip=list(identity)
                    )
                difference = values["duration_simulation"] - observed_duration
                count += 1
                sum_real += observed_duration
                sum_real_sq += observed_duration**2
                sum_diff += difference
                sum_diff_sq += difference**2
    except (ValueError, TypeError, OSError, OverflowError) as exc:
        return _failure("csv_parse_error", "output_parse_failure", error=str(exc))
    if count != len(expected):
        return _failure(
            "missing_completed_trips",
            "demand_conservation",
            candidate_rows=count,
            completed_trips=len(expected),
        )
    denominator = sum_real_sq - sum_real * sum_real / count
    metrics = {
        "R2": 1 - sum_diff_sq / denominator if denominator > 0 else 0.0,
        "RMSE": math.sqrt(sum_diff_sq / count),
        "Total passengers": count,
        "std(sim-real)": math.sqrt(max(0.0, sum_diff_sq / count - (sum_diff / count) ** 2)),
    }
    for key, computed in metrics.items():
        tolerance = REPORT_R2_TOL if key == "R2" else REPORT_METRIC_TOL
        if key == "Total passengers":
            tolerance = 0
        if abs(report[key] - computed) > tolerance:
            return _failure(
                "report_mismatch",
                "validation_report_metrics",
                metric=key,
                reported=report[key],
                recomputed=computed,
            )
    return ScoreResult(
        1.0,
        True,
        "ok",
        None,
        {
            "contract_version": CONTRACT_VERSION,
            "seed": seed,
            **simulation.diagnostics,
            "coverage": count / len(simulation.trips),
            "metrics": metrics,
            "historical_reference_used": False,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    args = parser.parse_args()
    result = score_output_bundle(
        output_dir=args.output_dir, reference_dir=args.reference_dir, input_dir=args.input_dir
    )
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
