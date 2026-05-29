#!/usr/bin/env python
"""Score web_admin_product_customer_cancel_v1 outputs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


TARGET_ORDER_ID = 302


@dataclass
class ScoreResult:
    score: float
    answer_ok: bool
    order_302_canceled: bool
    products_ok: bool
    customers_ok: bool
    non_target_orders_ok: bool
    target_order_ok: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_single_line(text: str | None) -> tuple[str | None, list[str]]:
    reasons: list[str] = []
    if text is None:
        reasons.append("missing final_answer.txt")
        return None, reasons
    lines = [line.rstrip("\n\r") for line in str(text).splitlines()]
    nonempty = [line for line in lines if line.strip()]
    if len(nonempty) != 1:
        reasons.append("final_answer.txt must contain exactly one non-empty line")
        return None, reasons
    return nonempty[0].strip(), reasons


def _orders_by_id(snapshot: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(order["id"]): order for order in snapshot.get("orders", [])}


def score_candidate(
    *,
    candidate_answer_text: str | None,
    candidate_state: dict[str, Any] | None,
    expected_answer_text: str,
    baseline_state: dict[str, Any],
    expected_state: dict[str, Any],
) -> ScoreResult:
    reasons: list[str] = []

    candidate_answer, answer_reasons = _normalize_single_line(candidate_answer_text)
    reasons.extend(answer_reasons)
    answer_ok = candidate_answer == expected_answer_text.strip()
    if candidate_answer is not None and not answer_ok:
        reasons.append("final answer does not exactly match the hidden reference")

    if candidate_state is None:
        reasons.append("missing candidate state snapshot")
        return ScoreResult(
            score=0.0,
            answer_ok=answer_ok,
            order_302_canceled=False,
            products_ok=False,
            customers_ok=False,
            non_target_orders_ok=False,
            target_order_ok=False,
            reasons=reasons,
        )

    products_ok = candidate_state.get("products") == baseline_state.get("products")
    customers_ok = candidate_state.get("customers") == baseline_state.get("customers")

    candidate_orders = _orders_by_id(candidate_state)
    baseline_orders = _orders_by_id(baseline_state)
    expected_orders = _orders_by_id(expected_state)

    target_order = candidate_orders.get(TARGET_ORDER_ID)
    expected_target_order = expected_orders.get(TARGET_ORDER_ID)
    order_302_canceled = bool(target_order and target_order.get("status") == "Canceled")
    target_order_ok = target_order == expected_target_order

    non_target_orders_ok = True
    baseline_ids = set(baseline_orders)
    if set(candidate_orders) != baseline_ids:
        non_target_orders_ok = False
    else:
        for order_id, baseline_order in baseline_orders.items():
            if order_id == TARGET_ORDER_ID:
                continue
            if candidate_orders.get(order_id) != baseline_order:
                non_target_orders_ok = False
                break

    if not products_ok:
        reasons.append("product records differ from the baseline state")
    if not customers_ok:
        reasons.append("customer records differ from the baseline state")
    if not order_302_canceled:
        reasons.append("order #302 is not canceled")
    if not non_target_orders_ok:
        reasons.append("one or more non-target orders differ from baseline")
    if not target_order_ok:
        reasons.append("order #302 does not match the expected post-cancel state")

    passed = all(
        [
            answer_ok,
            products_ok,
            customers_ok,
            order_302_canceled,
            non_target_orders_ok,
            target_order_ok,
        ]
    )
    return ScoreResult(
        score=1.0 if passed else 0.0,
        answer_ok=answer_ok,
        order_302_canceled=order_302_canceled,
        products_ok=products_ok,
        customers_ok=customers_ok,
        non_target_orders_ok=non_target_orders_ok,
        target_order_ok=target_order_ok,
        reasons=reasons,
    )


def _read_text(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.exists() else None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = score_candidate(
        candidate_answer_text=_read_text(args.candidate_dir / "final_answer.txt"),
        candidate_state=_read_json(args.candidate_dir / "state_snapshot.json")
        if (args.candidate_dir / "state_snapshot.json").exists()
        else None,
        expected_answer_text=_read_text(args.reference_dir / "reference_answer.txt") or "",
        baseline_state=_read_json(args.reference_dir / "baseline_state.json"),
        expected_state=_read_json(args.reference_dir / "expected_state_after_cancel.json"),
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
