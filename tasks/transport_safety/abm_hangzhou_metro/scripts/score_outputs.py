"""Scoring helpers for transport_safety/abm_hangzhou_metro."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

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
VISIBLE_COLUMNS = ["card_id", "o_station", "d_station", "start_time", "end_time"]
REPORT_KEYS = ["R2", "RMSE", "Total passengers", "std(sim-real)"]

COVERAGE_FLOOR = 0.80
ANTI_COPY_EXACT_MARGIN = 0.03
ANTI_COPY_WITHIN1_MARGIN = 0.05
ANTI_COPY_MAE_MARGIN = 0.30
REFERENCE_OVERLAP_MIN = 0.95
HIDDEN_END_MAE_MAX = 4.0
HIDDEN_DURATION_MAE_MAX = 4.0
TRANSFER_EXACT_RATE_MIN = 0.70
REPORT_R2_TOL = 0.001
REPORT_METRIC_TOL = 0.02


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
    return ScoreResult(score=0.0, passed=False, reason=reason, hard_gate=hard_gate, details=details)


def _parse_int_like(value: str, *, field: str) -> int:
    text = value.strip()
    if not text:
        raise ValueError(f"empty integer field: {field}")
    if "." in text or "e" in text.lower():
        parsed = float(text)
        if not math.isfinite(parsed) or not parsed.is_integer():
            raise ValueError(f"non-integral numeric field {field}: {value!r}")
        return int(parsed)
    return int(text)


def _duration_minutes(start_time: int, end_time: int) -> int:
    duration = end_time - start_time
    if duration < 0:
        duration += 1440
    return duration


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-200000")
    conn.execute(
        """
        CREATE TABLE visible (
            card_id TEXT NOT NULL,
            o_station TEXT NOT NULL,
            d_station TEXT NOT NULL,
            start_time INTEGER NOT NULL,
            end_time INTEGER NOT NULL,
            duration_real INTEGER NOT NULL,
            PRIMARY KEY (card_id, o_station, d_station, start_time)
        ) WITHOUT ROWID
        """
    )
    conn.execute(
        """
        CREATE TABLE reference_rows (
            card_id TEXT NOT NULL,
            o_station TEXT NOT NULL,
            d_station TEXT NOT NULL,
            start_time INTEGER NOT NULL,
            end_time_simulate INTEGER NOT NULL,
            transfer_times INTEGER NOT NULL,
            end_time_real INTEGER NOT NULL,
            duration_simulation INTEGER NOT NULL,
            duration_real INTEGER NOT NULL,
            PRIMARY KEY (card_id, o_station, d_station, start_time)
        ) WITHOUT ROWID
        """
    )
    conn.execute(
        """
        CREATE TABLE candidate_rows (
            card_id TEXT NOT NULL,
            o_station TEXT NOT NULL,
            d_station TEXT NOT NULL,
            start_time INTEGER NOT NULL,
            end_time_simulate INTEGER NOT NULL,
            transfer_times INTEGER NOT NULL,
            end_time_real INTEGER NOT NULL,
            duration_simulation INTEGER NOT NULL,
            duration_real INTEGER NOT NULL,
            PRIMARY KEY (card_id, o_station, d_station, start_time)
        ) WITHOUT ROWID
        """
    )
    return conn


def _load_visible_input(conn: sqlite3.Connection, csv_path: Path) -> int:
    row_count = 0
    batch: list[tuple[object, ...]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        observed = reader.fieldnames or []
        if [name.strip() for name in observed] != VISIBLE_COLUMNS:
            raise ValueError(f"unexpected visible input header: {observed}")
        for row in reader:
            start_time = _parse_int_like(row["start_time"], field="start_time")
            end_time = _parse_int_like(row["end_time"], field="end_time")
            batch.append(
                (
                    row["card_id"],
                    row["o_station"],
                    row["d_station"],
                    start_time,
                    end_time,
                    _duration_minutes(start_time, end_time),
                )
            )
            row_count += 1
            if len(batch) >= 10000:
                conn.executemany("INSERT INTO visible VALUES (?, ?, ?, ?, ?, ?)", batch)
                batch.clear()
    if batch:
        conn.executemany("INSERT INTO visible VALUES (?, ?, ?, ?, ?, ?)", batch)
    conn.commit()
    return row_count


def _load_output_csv(conn: sqlite3.Connection, csv_path: Path, table_name: str) -> int:
    row_count = 0
    batch: list[tuple[object, ...]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        observed = [name.strip() for name in (reader.fieldnames or [])]
        if observed != OUTPUT_COLUMNS:
            raise ValueError(f"unexpected output header in {csv_path.name}: {observed}")
        for row in reader:
            batch.append(
                (
                    row["card_id"],
                    row["o_station"],
                    row["d_station"],
                    _parse_int_like(row["start_time"], field="start_time"),
                    _parse_int_like(row["end_time_simulate"], field="end_time_simulate"),
                    _parse_int_like(row["transfer_times"], field="transfer_times"),
                    _parse_int_like(row["end_time_real"], field="end_time_real"),
                    _parse_int_like(row["duration_simulation"], field="duration_simulation"),
                    _parse_int_like(row["duration_real"], field="duration_real"),
                )
            )
            row_count += 1
            if len(batch) >= 10000:
                conn.executemany(f"INSERT INTO {table_name} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", batch)
                batch.clear()
    if batch:
        conn.executemany(f"INSERT INTO {table_name} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", batch)
    conn.commit()
    return row_count


def _parse_validation_report(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, raw_value = [part.strip() for part in line.split("=", 1)]
        if key == "RMSE":
            values[key] = float(raw_value.removesuffix(" minutes").strip())
        elif key == "Total passengers":
            values[key] = float(int(raw_value))
        elif key in {"R2", "std(sim-real)"}:
            values[key] = float(raw_value)
    missing = [key for key in REPORT_KEYS if key not in values]
    if missing:
        raise ValueError(f"validation report missing keys: {missing}")
    return values


def _fetch_one(conn: sqlite3.Connection, query: str) -> tuple[object, ...]:
    row = conn.execute(query).fetchone()
    if row is None:
        raise RuntimeError(f"query returned no rows: {query}")
    return row


def score_output_bundle(
    *,
    output_dir: Path,
    reference_dir: Path,
    input_dir: Path,
) -> ScoreResult:
    candidate_csv = output_dir / "passenger_records.csv"
    candidate_report = output_dir / "validation_report.txt"
    reference_csv = reference_dir / "passenger_records.csv"
    visible_csv = input_dir / "data" / "afc_hangzhou.csv"

    for path, label in [
        (candidate_csv, "candidate passenger_records.csv"),
        (candidate_report, "candidate validation_report.txt"),
        (reference_csv, "reference passenger_records.csv"),
        (visible_csv, "visible afc_hangzhou.csv"),
    ]:
        if not path.exists():
            return _failure("missing_file", "missing_required_file", missing=label, path=str(path))

    try:
        report_values = _parse_validation_report(candidate_report)
    except Exception as exc:
        return _failure("report_parse_error", "validation_report_parse_failure", error=str(exc))

    with tempfile.TemporaryDirectory(prefix="abm_hangzhou_metro_eval_") as tmp_dir:
        db_path = Path(tmp_dir) / "score.sqlite3"
        conn = _connect(db_path)
        try:
            try:
                visible_count = _load_visible_input(conn, visible_csv)
                reference_count = _load_output_csv(conn, reference_csv, "reference_rows")
                candidate_count = _load_output_csv(conn, candidate_csv, "candidate_rows")
            except sqlite3.IntegrityError as exc:
                return _failure(
                    "duplicate_trip_key",
                    "non_unique_trip_identity",
                    error=str(exc),
                    trip_identity_key=list(TRIP_KEY),
                )
            except Exception as exc:
                return _failure("csv_parse_error", "output_parse_failure", error=str(exc))

            coverage = candidate_count / visible_count if visible_count else 0.0
            if coverage < COVERAGE_FLOOR:
                return _failure(
                    "insufficient_coverage",
                    "coverage_floor",
                    candidate_rows=candidate_count,
                    visible_rows=visible_count,
                    coverage=coverage,
                )

            missing_visible = _fetch_one(
                conn,
                """
                SELECT COUNT(*)
                FROM candidate_rows c
                LEFT JOIN visible v
                  ON c.card_id = v.card_id
                 AND c.o_station = v.o_station
                 AND c.d_station = v.d_station
                 AND c.start_time = v.start_time
                WHERE v.card_id IS NULL
                """,
            )[0]
            if int(missing_visible):
                return _failure(
                    "unknown_trip_keys",
                    "candidate_not_subset_of_visible_input",
                    missing_visible_rows=int(missing_visible),
                )

            (
                joined_visible_count,
                real_endtime_match_rate,
                real_duration_match_rate,
                exact_visible_duration,
                exact_visible_endtime,
                mae_visible_duration,
                within1_visible_duration,
                within1_visible_endtime,
                report_sse,
                report_sum_real,
                report_sum_real_sq,
                report_sum_diff,
                report_sum_diff_sq,
            ) = _fetch_one(
                conn,
                """
                SELECT
                    COUNT(*),
                    AVG(CASE WHEN c.end_time_real = v.end_time THEN 1.0 ELSE 0.0 END),
                    AVG(CASE WHEN c.duration_real = v.duration_real THEN 1.0 ELSE 0.0 END),
                    AVG(CASE WHEN c.duration_simulation = v.duration_real THEN 1.0 ELSE 0.0 END),
                    AVG(CASE WHEN c.end_time_simulate = v.end_time THEN 1.0 ELSE 0.0 END),
                    AVG(ABS(c.duration_simulation - v.duration_real)),
                    AVG(CASE WHEN ABS(c.duration_simulation - v.duration_real) <= 1 THEN 1.0 ELSE 0.0 END),
                    AVG(CASE WHEN ABS(c.end_time_simulate - v.end_time) <= 1 THEN 1.0 ELSE 0.0 END),
                    SUM((c.duration_simulation - v.duration_real) * (c.duration_simulation - v.duration_real)),
                    SUM(v.duration_real),
                    SUM(v.duration_real * v.duration_real),
                    SUM(c.duration_simulation - v.duration_real),
                    SUM((c.duration_simulation - v.duration_real) * (c.duration_simulation - v.duration_real))
                FROM candidate_rows c
                JOIN visible v
                  ON c.card_id = v.card_id
                 AND c.o_station = v.o_station
                 AND c.d_station = v.d_station
                 AND c.start_time = v.start_time
                """,
            )
            joined_visible_count = int(joined_visible_count)
            if joined_visible_count != candidate_count:
                return _failure(
                    "visible_join_mismatch",
                    "candidate_visible_join_count",
                    joined_visible_rows=joined_visible_count,
                    candidate_rows=candidate_count,
                )
            (
                reference_exact_visible_duration,
                reference_exact_visible_endtime,
                reference_mae_visible_duration,
                reference_within1_visible_duration,
                reference_within1_visible_endtime,
            ) = _fetch_one(
                conn,
                """
                SELECT
                    AVG(CASE WHEN r.duration_simulation = v.duration_real THEN 1.0 ELSE 0.0 END),
                    AVG(CASE WHEN r.end_time_simulate = v.end_time THEN 1.0 ELSE 0.0 END),
                    AVG(ABS(r.duration_simulation - v.duration_real)),
                    AVG(CASE WHEN ABS(r.duration_simulation - v.duration_real) <= 1 THEN 1.0 ELSE 0.0 END),
                    AVG(CASE WHEN ABS(r.end_time_simulate - v.end_time) <= 1 THEN 1.0 ELSE 0.0 END)
                FROM reference_rows r
                JOIN visible v
                  ON r.card_id = v.card_id
                 AND r.o_station = v.o_station
                 AND r.d_station = v.d_station
                 AND r.start_time = v.start_time
                """,
            )
            if float(real_endtime_match_rate) < 1.0 or float(real_duration_match_rate) < 1.0:
                return _failure(
                    "visible_label_echo_mismatch",
                    "output_contract_real_columns",
                    real_endtime_match_rate=float(real_endtime_match_rate),
                    real_duration_match_rate=float(real_duration_match_rate),
                )

            if (
                float(exact_visible_duration) > float(reference_exact_visible_duration) + ANTI_COPY_EXACT_MARGIN
                or float(exact_visible_endtime) > float(reference_exact_visible_endtime) + ANTI_COPY_EXACT_MARGIN
                or float(within1_visible_duration)
                > float(reference_within1_visible_duration) + ANTI_COPY_WITHIN1_MARGIN
                or float(within1_visible_endtime)
                > float(reference_within1_visible_endtime) + ANTI_COPY_WITHIN1_MARGIN
                or float(mae_visible_duration) < float(reference_mae_visible_duration) - ANTI_COPY_MAE_MARGIN
            ):
                return _failure(
                    "visible_label_copy",
                    "anti_copy_guard",
                    reference_duration_mae_to_visible=float(reference_mae_visible_duration),
                    duration_mae_to_visible=float(mae_visible_duration),
                    reference_exact_duration_copy_rate=float(reference_exact_visible_duration),
                    exact_duration_copy_rate=float(exact_visible_duration),
                    reference_exact_endtime_copy_rate=float(reference_exact_visible_endtime),
                    exact_endtime_copy_rate=float(exact_visible_endtime),
                    reference_duration_within_1min_rate=float(reference_within1_visible_duration),
                    duration_within_1min_rate=float(within1_visible_duration),
                    reference_endtime_within_1min_rate=float(reference_within1_visible_endtime),
                    endtime_within_1min_rate=float(within1_visible_endtime),
                )

            (
                overlap_count,
                mae_hidden_end,
                mae_hidden_duration,
                transfer_exact_rate,
            ) = _fetch_one(
                conn,
                """
                SELECT
                    COUNT(*),
                    AVG(ABS(c.end_time_simulate - r.end_time_simulate)),
                    AVG(ABS(c.duration_simulation - r.duration_simulation)),
                    AVG(CASE WHEN c.transfer_times = r.transfer_times THEN 1.0 ELSE 0.0 END)
                FROM candidate_rows c
                JOIN reference_rows r
                  ON c.card_id = r.card_id
                 AND c.o_station = r.o_station
                 AND c.d_station = r.d_station
                 AND c.start_time = r.start_time
                """,
            )
            overlap_count = int(overlap_count)
            reference_overlap = overlap_count / reference_count if reference_count else 0.0
            if reference_overlap < REFERENCE_OVERLAP_MIN:
                return _failure(
                    "insufficient_reference_overlap",
                    "reference_overlap_floor",
                    overlap_rows=overlap_count,
                    reference_rows=reference_count,
                    overlap_fraction=reference_overlap,
                )

            if (
                float(mae_hidden_end) > HIDDEN_END_MAE_MAX
                or float(mae_hidden_duration) > HIDDEN_DURATION_MAE_MAX
                or float(transfer_exact_rate) < TRANSFER_EXACT_RATE_MIN
            ):
                return _failure(
                    "hidden_reference_mismatch",
                    "hidden_reference_thresholds",
                    overlap_rows=overlap_count,
                    overlap_fraction=reference_overlap,
                    mae_hidden_end=float(mae_hidden_end),
                    mae_hidden_duration=float(mae_hidden_duration),
                    transfer_exact_rate=float(transfer_exact_rate),
                )

            report_sse = float(report_sse)
            report_sum_real = float(report_sum_real)
            report_sum_real_sq = float(report_sum_real_sq)
            report_sum_diff = float(report_sum_diff)
            report_sum_diff_sq = float(report_sum_diff_sq)
            report_mean_real = report_sum_real / joined_visible_count
            report_sst = report_sum_real_sq - joined_visible_count * report_mean_real * report_mean_real
            recomputed_r2 = 1.0 - (report_sse / report_sst) if report_sst > 0 else 0.0
            recomputed_rmse = math.sqrt(report_sse / joined_visible_count)
            recomputed_std = math.sqrt(
                max(0.0, report_sum_diff_sq / joined_visible_count - (report_sum_diff / joined_visible_count) ** 2)
            )

            if int(report_values["Total passengers"]) != candidate_count:
                return _failure(
                    "report_mismatch",
                    "validation_report_total_passengers",
                    reported_total=int(report_values["Total passengers"]),
                    candidate_rows=candidate_count,
                )
            if abs(report_values["R2"] - recomputed_r2) > REPORT_R2_TOL:
                return _failure(
                    "report_mismatch",
                    "validation_report_r2",
                    reported=report_values["R2"],
                    recomputed=recomputed_r2,
                )
            for key, recomputed in [("RMSE", recomputed_rmse), ("std(sim-real)", recomputed_std)]:
                if abs(report_values[key] - recomputed) > REPORT_METRIC_TOL:
                    return _failure(
                        "report_mismatch",
                        f"validation_report_{key}",
                        reported=report_values[key],
                        recomputed=recomputed,
                    )

            return ScoreResult(
                score=1.0,
                passed=True,
                reason="ok",
                hard_gate=None,
                details={
                    "visible_rows": visible_count,
                    "candidate_rows": candidate_count,
                    "reference_rows": reference_count,
                    "coverage": coverage,
                    "reference_overlap_fraction": reference_overlap,
                    "real_endtime_match_rate": float(real_endtime_match_rate),
                    "real_duration_match_rate": float(real_duration_match_rate),
                    "reference_duration_mae_to_visible": float(reference_mae_visible_duration),
                    "duration_mae_to_visible": float(mae_visible_duration),
                    "reference_exact_duration_copy_rate": float(reference_exact_visible_duration),
                    "exact_duration_copy_rate": float(exact_visible_duration),
                    "reference_exact_endtime_copy_rate": float(reference_exact_visible_endtime),
                    "exact_endtime_copy_rate": float(exact_visible_endtime),
                    "reference_duration_within_1min_rate": float(reference_within1_visible_duration),
                    "duration_within_1min_rate": float(within1_visible_duration),
                    "reference_endtime_within_1min_rate": float(reference_within1_visible_endtime),
                    "endtime_within_1min_rate": float(within1_visible_endtime),
                    "mae_hidden_end": float(mae_hidden_end),
                    "mae_hidden_duration": float(mae_hidden_duration),
                    "transfer_exact_rate": float(transfer_exact_rate),
                    "reported_r2": report_values["R2"],
                    "recomputed_r2": recomputed_r2,
                    "reported_rmse": report_values["RMSE"],
                    "recomputed_rmse": recomputed_rmse,
                    "reported_std_sim_minus_real": report_values["std(sim-real)"],
                    "recomputed_std_sim_minus_real": recomputed_std,
                },
            )
        finally:
            conn.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args()

    result = score_output_bundle(
        output_dir=Path(args.output_dir),
        reference_dir=Path(args.reference_dir),
        input_dir=Path(args.input_dir),
    )
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
