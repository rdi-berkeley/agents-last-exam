"""Deterministic scorer for data_pipeline_etl_instance_1."""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any


REQUIRED_FILES = {
    "warehouse.db",
    "data_quality_report.json",
    "warehouse_summary.json",
}

EXPECTED_TABLE_COLUMNS = {
    "fact_transactions": {
        "transaction_id",
        "customer_key",
        "product_key",
        "date_key",
        "quantity",
        "unit_price",
        "total_amount",
        "discount_pct",
        "channel",
    },
    "dim_customers": {
        "customer_key",
        "customer_id",
        "name",
        "email",
        "segment",
        "country_code",
        "registration_date",
    },
    "dim_products": {
        "product_key",
        "product_id",
        "product_name",
        "category",
        "subcategory",
        "base_price",
        "cost_price",
        "supplier",
        "is_active",
    },
    "dim_dates": {
        "date_key",
        "full_date",
        "year",
        "month",
        "day",
        "day_of_week",
        "is_weekend",
        "quarter",
    },
}

VALID_SEGMENTS = {"basic", "premium", "standard", "unknown"}
VALID_COUNTRY_CODES = {"US", "GB", "CA", "DE", "XX"}
VALID_SUPPLIERS = {"Supplier A", "Supplier B", "Supplier C", "Supplier D"}
RAW_INPUT_COUNTS = {
    "raw_transaction_files": 6,
    "raw_transaction_rows": 6727,
    "raw_customer_records": 519,
    "raw_product_records": 55,
}
RAW_TRANSFORMATION_COUNTS = {
    "duplicates_removed": 197,
    "null_rows_dropped": 616,
    "null_customer_ids_mapped": 256,
}
EXPECTED_GHOST_CUSTOMER_IDS = list(range(501, 520))

REQUIRED_REPORT_KEYS = {
    "extraction": [
        "raw_transaction_files",
        "raw_transaction_rows",
        "raw_customer_records",
        "raw_product_records",
    ],
    "transformations": [
        "duplicates_removed",
        "null_rows_dropped",
        "null_customer_ids_mapped",
        "timestamps_standardized",
        "schema_drift_columns_filled",
        "country_codes_standardized",
        "supplier_names_standardized",
        "boolean_fields_normalized",
        "empty_categories_labeled",
    ],
    "load": [
        "fact_transactions_rows",
        "dim_customers_rows",
        "dim_products_rows",
        "dim_dates_rows",
        "referential_integrity_violations_remaining",
    ],
    "quality_checks": [
        "no_duplicate_transaction_ids",
        "no_null_quantities",
        "no_null_unit_prices",
        "total_amount_formula_correct",
        "all_foreign_keys_resolve",
        "all_segments_valid",
        "all_country_codes_iso2",
    ],
}

REQUIRED_SUMMARY_KEYS = {
    "fact_table": [
        "row_count",
        "unique_transactions",
        "unique_customers",
        "unique_products",
        "total_revenue",
        "avg_transaction_value",
        "date_range",
        "channel_distribution",
    ],
    "dim_customers": [
        "row_count",
        "segment_distribution",
        "country_distribution",
    ],
    "dim_products": [
        "row_count",
        "category_distribution",
        "active_products",
    ],
}


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def sqlite_tables(conn: sqlite3.Connection) -> dict[str, set[str]]:
    tables = [
        row[0]
        for row in conn.execute(
            "select name from sqlite_master where type='table' and name not like 'sqlite_%' order by name"
        ).fetchall()
    ]
    output: dict[str, set[str]] = {}
    for table in tables:
        cols = {row[1] for row in conn.execute(f"pragma table_info({table})").fetchall()}
        output[table] = cols
    return output


def counts_by_value(conn: sqlite3.Connection, sql: str) -> dict[str, int]:
    return {str(key): int(count) for key, count in conn.execute(sql).fetchall()}


def parse_date_text(raw: str) -> date:
    token = raw.strip()
    try:
        return date.fromisoformat(token)
    except ValueError:
        pass

    normalized = token.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).date()
    except ValueError as exc:
        raise ValueError(f"Unsupported date value: {raw!r}") from exc


def warehouse_metrics(db_path: Path) -> dict[str, Any]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        fact_rows = int(conn.execute("select count(*) from fact_transactions").fetchone()[0])
        distinct_txn = int(conn.execute("select count(distinct transaction_id) from fact_transactions").fetchone()[0])
        customer_rows = int(conn.execute("select count(*) from dim_customers").fetchone()[0])
        product_rows = int(conn.execute("select count(*) from dim_products").fetchone()[0])
        date_rows = int(conn.execute("select count(*) from dim_dates").fetchone()[0])
        unique_customers = int(conn.execute("select count(distinct customer_key) from fact_transactions").fetchone()[0])
        unique_products = int(conn.execute("select count(distinct product_key) from fact_transactions").fetchone()[0])
        total_revenue = float(conn.execute("select round(sum(total_amount), 2) from fact_transactions").fetchone()[0] or 0.0)
        avg_transaction = float(conn.execute("select round(avg(total_amount), 2) from fact_transactions").fetchone()[0] or 0.0)
        date_min, date_max = conn.execute("select min(date_key), max(date_key) from fact_transactions").fetchone()
        unknown_customer_rows = int(conn.execute("select count(*) from dim_customers where customer_id = 0").fetchone()[0])
        unknown_customer_fact_rows = int(
            conn.execute(
                """
                select count(*)
                from fact_transactions f
                join dim_customers c on c.customer_key = f.customer_key
                where c.customer_id = 0
                """
            ).fetchone()[0]
        )
        ghost_customer_ids = [
            int(row[0])
            for row in conn.execute(
                """
                select customer_id
                from dim_customers
                where customer_id between 501 and 519
                order by customer_id
                """
            ).fetchall()
        ]
        unresolved_fk = int(
            conn.execute(
                """
                select count(*)
                from fact_transactions f
                left join dim_customers c on c.customer_key = f.customer_key
                left join dim_products p on p.product_key = f.product_key
                left join dim_dates d on d.date_key = f.date_key
                where c.customer_key is null or p.product_key is null or d.date_key is null
                """
            ).fetchone()[0]
        )
        formula_violations = int(
            conn.execute(
                """
                select count(*)
                from fact_transactions
                where abs(total_amount - (quantity * unit_price * (1 - discount_pct))) > 0.01
                """
            ).fetchone()[0]
        )
        null_quantity = int(conn.execute("select count(*) from fact_transactions where quantity is null").fetchone()[0])
        null_unit_price = int(conn.execute("select count(*) from fact_transactions where unit_price is null").fetchone()[0])
        invalid_segments = int(
            conn.execute(
                "select count(*) from dim_customers where segment is null or segment not in ('basic','premium','standard','unknown')"
            ).fetchone()[0]
        )
        invalid_country_codes = int(
            conn.execute(
                """
                select count(*)
                from dim_customers
                where country_code is null
                   or length(country_code) != 2
                   or country_code not in ('US','GB','CA','DE','XX')
                """
            ).fetchone()[0]
        )
        invalid_supplier_rows = int(
            conn.execute(
                """
                select count(*)
                from dim_products
                where supplier is not null
                  and supplier not in ('Supplier A','Supplier B','Supplier C','Supplier D')
                """
            ).fetchone()[0]
        )
        invalid_boolean_rows = int(conn.execute("select count(*) from dim_products where is_active not in (0, 1)").fetchone()[0])
        null_category_rows = int(conn.execute("select count(*) from dim_products where category is null").fetchone()[0])
        channel_distribution = counts_by_value(
            conn,
            "select coalesce(channel, '<missing>'), count(*) from fact_transactions group by 1",
        )
        segment_distribution = counts_by_value(
            conn,
            "select coalesce(segment, '<missing>'), count(*) from dim_customers group by 1",
        )
        country_distribution = counts_by_value(
            conn,
            "select coalesce(country_code, '<missing>'), count(*) from dim_customers group by 1",
        )
        category_distribution = counts_by_value(
            conn,
            "select coalesce(category, '<missing>'), count(*) from dim_products group by 1",
        )
        active_products = int(conn.execute("select count(*) from dim_products where is_active = 1").fetchone()[0])
        date_rows_payload = conn.execute(
            """
            select date_key, full_date, year, month, day, day_of_week, is_weekend, quarter
            from dim_dates
            order by date_key
            """
        ).fetchall()

    expected_start = date(2024, 1, 1)
    date_dimension_valid = len(date_rows_payload) == 182
    if date_dimension_valid:
        for idx, row in enumerate(date_rows_payload):
            expected = date.fromordinal(expected_start.toordinal() + idx)
            try:
                full_date = parse_date_text(row["full_date"])
            except Exception:
                date_dimension_valid = False
                break
            expected_day_name = expected.strftime("%A")
            expected_quarter = ((expected.month - 1) // 3) + 1
            expected_weekend = 1 if expected.weekday() >= 5 else 0
            if (
                row["date_key"] != int(expected.strftime("%Y%m%d"))
                or full_date != expected
                or row["year"] != expected.year
                or row["month"] != expected.month
                or row["day"] != expected.day
                or row["day_of_week"] != expected_day_name
                or int(row["is_weekend"]) != expected_weekend
                or row["quarter"] != expected_quarter
            ):
                date_dimension_valid = False
                break

    unknown_customer_contract_valid = unknown_customer_rows == 1 and unknown_customer_fact_rows > 0
    ghost_customer_contract_valid = ghost_customer_ids == EXPECTED_GHOST_CUSTOMER_IDS

    return {
        "fact_rows": fact_rows,
        "distinct_txn": distinct_txn,
        "customer_rows": customer_rows,
        "product_rows": product_rows,
        "date_rows": date_rows,
        "unique_customers": unique_customers,
        "unique_products": unique_products,
        "total_revenue": total_revenue,
        "avg_transaction_value": avg_transaction,
        "date_min": int(date_min),
        "date_max": int(date_max),
        "unknown_customer_rows": unknown_customer_rows,
        "unknown_customer_fact_rows": unknown_customer_fact_rows,
        "ghost_customer_ids": ghost_customer_ids,
        "date_dimension_valid": date_dimension_valid,
        "unknown_customer_contract_valid": unknown_customer_contract_valid,
        "ghost_customer_contract_valid": ghost_customer_contract_valid,
        "unresolved_fk": unresolved_fk,
        "formula_violations": formula_violations,
        "null_quantity": null_quantity,
        "null_unit_price": null_unit_price,
        "invalid_segments": invalid_segments,
        "invalid_country_codes": invalid_country_codes,
        "invalid_supplier_rows": invalid_supplier_rows,
        "invalid_boolean_rows": invalid_boolean_rows,
        "null_category_rows": null_category_rows,
        "channel_distribution": channel_distribution,
        "segment_distribution": segment_distribution,
        "country_distribution": country_distribution,
        "category_distribution": category_distribution,
        "active_products": active_products,
    }


def within_pct(actual: int | float, target: int | float, pct: float) -> bool:
    return abs(actual - target) <= abs(target) * pct


def schema_correct(candidate_db: Path) -> bool:
    with sqlite3.connect(candidate_db) as conn:
        tables = sqlite_tables(conn)
    if set(tables) != set(EXPECTED_TABLE_COLUMNS):
        return False
    return all(tables[table] == EXPECTED_TABLE_COLUMNS[table] for table in EXPECTED_TABLE_COLUMNS)


def report_and_summary_truthful(
    report: dict[str, Any],
    summary: dict[str, Any],
    metrics: dict[str, Any],
) -> bool:
    for section, keys in REQUIRED_REPORT_KEYS.items():
        payload = report.get(section)
        if not isinstance(payload, dict):
            return False
        if any(key not in payload for key in keys):
            return False

    for section, keys in REQUIRED_SUMMARY_KEYS.items():
        payload = summary.get(section)
        if not isinstance(payload, dict):
            return False
        if any(key not in payload for key in keys):
            return False

    date_range = summary["fact_table"].get("date_range")
    if not isinstance(date_range, dict) or any(k not in date_range for k in ["min", "max"]):
        return False

    expected_quality = {
        "no_duplicate_transaction_ids": metrics["fact_rows"] == metrics["distinct_txn"],
        "no_null_quantities": metrics["null_quantity"] == 0,
        "no_null_unit_prices": metrics["null_unit_price"] == 0,
        "total_amount_formula_correct": metrics["formula_violations"] == 0,
        "all_foreign_keys_resolve": metrics["unresolved_fk"] == 0,
        "all_segments_valid": metrics["invalid_segments"] == 0,
        "all_country_codes_iso2": metrics["invalid_country_codes"] == 0,
    }
    if any(report["quality_checks"].get(k) != v for k, v in expected_quality.items()):
        return False

    if report["extraction"] != RAW_INPUT_COUNTS:
        return False

    transformations = report["transformations"]
    if transformations.get("duplicates_removed") != RAW_TRANSFORMATION_COUNTS["duplicates_removed"]:
        return False
    if transformations.get("null_rows_dropped") != RAW_TRANSFORMATION_COUNTS["null_rows_dropped"]:
        return False
    if transformations.get("null_customer_ids_mapped") != RAW_TRANSFORMATION_COUNTS["null_customer_ids_mapped"]:
        return False
    if (
        transformations["duplicates_removed"]
        + transformations["null_rows_dropped"]
        + metrics["fact_rows"]
        != RAW_INPUT_COUNTS["raw_transaction_rows"]
    ):
        return False
    if transformations.get("timestamps_standardized") is not True:
        return False
    if transformations.get("schema_drift_columns_filled") != ["discount_pct", "channel"]:
        return False
    if transformations.get("country_codes_standardized") != (metrics["invalid_country_codes"] == 0):
        return False
    if transformations.get("supplier_names_standardized") != (metrics["invalid_supplier_rows"] == 0):
        return False
    if transformations.get("boolean_fields_normalized") != (metrics["invalid_boolean_rows"] == 0):
        return False
    if transformations.get("empty_categories_labeled") != (metrics["null_category_rows"] == 0):
        return False

    expected_load = {
        "fact_transactions_rows": metrics["fact_rows"],
        "dim_customers_rows": metrics["customer_rows"],
        "dim_products_rows": metrics["product_rows"],
        "dim_dates_rows": metrics["date_rows"],
        "referential_integrity_violations_remaining": metrics["unresolved_fk"],
    }
    if any(report["load"].get(k) != v for k, v in expected_load.items()):
        return False

    expected_summary = {
        ("fact_table", "row_count"): metrics["fact_rows"],
        ("fact_table", "unique_transactions"): metrics["distinct_txn"],
        ("fact_table", "unique_customers"): metrics["unique_customers"],
        ("fact_table", "unique_products"): metrics["unique_products"],
        ("fact_table", "total_revenue"): metrics["total_revenue"],
        ("fact_table", "avg_transaction_value"): metrics["avg_transaction_value"],
        ("dim_customers", "row_count"): metrics["customer_rows"],
        ("dim_products", "row_count"): metrics["product_rows"],
        ("dim_products", "active_products"): metrics["active_products"],
    }
    for (section, key), value in expected_summary.items():
        if summary[section].get(key) != value:
            return False

    if summary["fact_table"]["date_range"].get("min") != metrics["date_min"]:
        return False
    if summary["fact_table"]["date_range"].get("max") != metrics["date_max"]:
        return False
    if summary["fact_table"].get("channel_distribution") != metrics["channel_distribution"]:
        return False
    if summary["dim_customers"].get("segment_distribution") != metrics["segment_distribution"]:
        return False
    if summary["dim_customers"].get("country_distribution") != metrics["country_distribution"]:
        return False
    if summary["dim_products"].get("category_distribution") != metrics["category_distribution"]:
        return False
    return True


def score_output_bundle(*, output_dir: Path, reference_db_path: Path) -> dict[str, Any]:
    missing = sorted(name for name in REQUIRED_FILES if not (output_dir / name).exists())
    if missing:
        return {"score": 0.0, "reason": f"missing required files: {missing}"}

    try:
        candidate_report = load_json(output_dir / "data_quality_report.json")
        candidate_summary = load_json(output_dir / "warehouse_summary.json")
        candidate_metrics = warehouse_metrics(output_dir / "warehouse.db")
        reference_metrics = warehouse_metrics(reference_db_path)
    except Exception as exc:
        return {"score": 0.0, "reason": f"failed to parse output bundle: {exc}"}

    criteria = {
        "schema_correct": (
            schema_correct(output_dir / "warehouse.db")
            and candidate_metrics["date_dimension_valid"]
            and candidate_metrics["unknown_customer_contract_valid"]
            and candidate_metrics["ghost_customer_contract_valid"]
        ),
        "row_counts_within_tolerance": (
            within_pct(candidate_metrics["fact_rows"], reference_metrics["fact_rows"], 0.05)
            and within_pct(candidate_metrics["customer_rows"], reference_metrics["customer_rows"], 0.05)
            and within_pct(candidate_metrics["product_rows"], reference_metrics["product_rows"], 0.05)
            and candidate_metrics["date_rows"] == reference_metrics["date_rows"]
        ),
        "data_quality_checks": (
            candidate_metrics["fact_rows"] == candidate_metrics["distinct_txn"]
            and candidate_metrics["null_quantity"] == 0
            and candidate_metrics["null_unit_price"] == 0
            and candidate_metrics["formula_violations"] == 0
            and candidate_metrics["unresolved_fk"] == 0
        ),
        "standardization_correct": (
            candidate_metrics["invalid_segments"] == 0
            and candidate_metrics["invalid_country_codes"] == 0
            and candidate_metrics["invalid_boolean_rows"] == 0
            and candidate_metrics["invalid_supplier_rows"] == 0
        ),
        "revenue_within_tolerance": within_pct(
            candidate_metrics["total_revenue"], reference_metrics["total_revenue"], 0.02
        ),
        "sidecars_truthful": report_and_summary_truthful(
            candidate_report,
            candidate_summary,
            candidate_metrics,
        ),
    }
    score = sum(1.0 for passed in criteria.values() if passed) / len(criteria)
    return {
        "score": score,
        "criteria": criteria,
        "candidate_metrics": candidate_metrics,
        "reference_metrics": {
            "fact_rows": reference_metrics["fact_rows"],
            "customer_rows": reference_metrics["customer_rows"],
            "product_rows": reference_metrics["product_rows"],
            "date_rows": reference_metrics["date_rows"],
            "total_revenue": reference_metrics["total_revenue"],
        },
    }
