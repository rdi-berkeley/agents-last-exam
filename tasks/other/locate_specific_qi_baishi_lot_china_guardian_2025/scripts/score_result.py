"""Scoring helpers for the China Guardian Qi Baishi lot lookup task."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


REQUIRED_KEYS = ("lot_number", "auction_date", "realized_price_cny", "source_url")


@dataclass(frozen=True)
class ScoreResult:
    score: float
    reasons: list[str]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _normalize_lot(value: Any) -> str:
    text = str(value or "").strip().upper()
    text = re.sub(r"^LOT\s*", "", text)
    return re.sub(r"\s+", "", text)


def _normalize_price(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(round(float(value)))
    text = str(value or "").strip()
    text = re.sub(r"(?i)\b(RMB|CNY|YUAN|CN¥|￥)\b", "", text)
    text = re.sub(r"[,\s]", "", text)
    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return None
    return int(round(float(match.group(0))))


def _normalize_datetime(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    text = text.replace("T", " ").replace("/", "-")
    text = re.sub(r"\s+", " ", text)
    formats = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%Y.%m.%d %H:%M:%S",
        "%Y.%m.%d %H:%M",
        "%Y.%m.%d",
    )
    for fmt in formats:
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        if fmt in ("%Y-%m-%d", "%Y.%m.%d"):
            return parsed.strftime("%Y-%m-%d")
        return parsed.strftime("%Y-%m-%d %H:%M:%S")
    return text


def _url_identity(value: Any) -> tuple[str, str, str] | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    parsed = urlparse(raw)
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    query = parse_qs(parsed.query)
    category = (query.get("categoryId") or query.get("categoryid") or [""])[0]
    item = (query.get("itemCode") or query.get("itemcode") or [""])[0]
    return host, category.strip(), item.strip()


def score_payload(agent: dict[str, Any], reference: dict[str, Any]) -> ScoreResult:
    reasons: list[str] = []
    missing = [key for key in REQUIRED_KEYS if key not in agent]
    if missing:
        return ScoreResult(0.0, [f"missing required key(s): {', '.join(missing)}"])

    score = 0.0

    if _normalize_lot(agent.get("lot_number")) == _normalize_lot(reference.get("lot_number")):
        score += 0.35
    else:
        reasons.append("lot_number mismatch")

    agent_url = _url_identity(agent.get("source_url"))
    reference_url = _url_identity(reference.get("source_url"))
    if agent_url and reference_url and agent_url == reference_url:
        score += 0.35
    else:
        reasons.append("source_url does not identify the reference China Guardian lot")

    agent_date = _normalize_datetime(agent.get("auction_date"))
    reference_date = _normalize_datetime(reference.get("auction_date"))
    if agent_date == reference_date:
        score += 0.10
    elif agent_date and reference_date and agent_date[:10] == reference_date[:10]:
        score += 0.05
        reasons.append("auction_date has correct date but missing or wrong time")
    else:
        reasons.append("auction_date mismatch")

    if _normalize_price(agent.get("realized_price_cny")) == _normalize_price(
        reference.get("realized_price_cny")
    ):
        score += 0.20
    else:
        reasons.append("realized_price_cny mismatch")

    return ScoreResult(round(score, 4), reasons)


def score_files(agent_path: Path, reference_path: Path) -> ScoreResult:
    try:
        agent = _load_json(agent_path)
    except Exception as exc:
        return ScoreResult(0.0, [f"invalid agent JSON: {exc}"])
    try:
        reference = _load_json(reference_path)
    except Exception as exc:
        return ScoreResult(0.0, [f"invalid reference JSON: {exc}"])
    return score_payload(agent, reference)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Score China Guardian lot result.json")
    parser.add_argument("--agent", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument(
        "--expect-score",
        type=float,
        help="Return success when the score matches this expected value.",
    )
    args = parser.parse_args()

    result = score_files(args.agent, args.reference)
    print(json.dumps({"score": result.score, "reasons": result.reasons}, indent=2))
    if args.expect_score is not None:
        return 0 if abs(result.score - args.expect_score) <= 1e-9 else 1
    return 0 if result.score >= 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
