#!/usr/bin/env python3
"""Canonical scorer for business_finance/sec_10k_financial_parsing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FINANCIAL_FIELDS = [
    "revenue",
    "cost_of_revenue",
    "operating_income",
    "net_income",
    "total_assets",
    "total_liabilities",
    "stockholders_equity",
    "cash_and_equivalents",
    "eps_basic",
    "eps_diluted",
]

METADATA_FIELDS = [
    "company_name",
    "ticker",
    "cik",
    "filing_date",
    "fiscal_year_end",
]

WEIGHTS = {
    "schema_compliance": 0.05,
    "metadata_accuracy": 0.05,
    "financial_accuracy": 0.30,
    "anls_verification": 0.10,
    "analytical_qa": 0.25,
    "cross_validation": 0.10,
    "completeness": 0.05,
    "determinism": 0.10,
}

ANLS_THRESHOLD = 0.6


def _load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _coerce_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _coerce_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if not text:
            return None
        try:
            if any(char in text for char in [".", "e", "E"]):
                parsed = float(text)
                return int(parsed) if parsed.is_integer() else None
            return int(text)
        except ValueError:
            return None
    return None


def _levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if not s2:
        return len(s1)
    previous = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            current.append(min(current[j] + 1, previous[j + 1] + 1, previous[j] + cost))
        previous = current
    return previous[-1]


def normalized_levenshtein_similarity(s1: str, s2: str) -> float:
    if not s1 and not s2:
        return 1.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    return 1.0 - (_levenshtein_distance(s1, s2) / max_len)


def compute_anls(raw_text: str, baseline_text: str, chunk_size: int = 500) -> float:
    if not raw_text.strip() or not baseline_text.strip():
        return 0.0
    raw_text = " ".join(raw_text.split())
    baseline_text = " ".join(baseline_text.split())
    if raw_text == baseline_text:
        return 1.0
    if raw_text in baseline_text or baseline_text in raw_text:
        shorter = min(len(raw_text), len(baseline_text))
        longer = max(len(raw_text), len(baseline_text))
        return shorter / longer if longer else 1.0

    chunks = []
    for i in range(0, len(raw_text), chunk_size):
        chunk = raw_text[i : i + chunk_size]
        if len(chunk) >= 50:
            chunks.append((i, chunk))
    if not chunks:
        return 0.0

    similarities: list[float] = []
    for start, chunk in chunks[:20]:
        if chunk in baseline_text:
            similarities.append(1.0)
            continue

        best = 0.0
        ratio = start / max(1, len(raw_text))
        center = int(ratio * len(baseline_text))
        offsets = [
            0,
            -(len(chunk) // 2),
            len(chunk) // 2,
            -len(chunk),
            len(chunk),
        ]
        for offset in offsets:
            window_start = max(0, min(len(baseline_text) - len(chunk), center + offset))
            window = baseline_text[window_start : window_start + len(chunk)]
            best = max(best, normalized_levenshtein_similarity(chunk, window))
            if best > 0.9:
                break
        similarities.append(best)
    return sum(similarities) / len(similarities)


def _schema_valid(payload: dict) -> bool:
    if not isinstance(payload, dict):
        return False
    if set(payload.keys()) != {"company_name", "ticker", "cik", "filing_date", "fiscal_year_end", "financials"}:
        return False
    if not all(isinstance(payload.get(field), str) for field in METADATA_FIELDS):
        return False
    financials = payload.get("financials")
    if not isinstance(financials, dict):
        return False
    if set(financials.keys()) != set(FINANCIAL_FIELDS):
        return False
    for field in FINANCIAL_FIELDS:
        value = financials[field]
        if value is None:
            continue
        if field.startswith("eps_"):
            if not isinstance(value, (int, float)):
                return False
        elif not isinstance(value, int):
            return False
    return True


def _metadata_score(prediction: dict, ground_truth: dict) -> tuple[int, int]:
    if not isinstance(prediction, dict) or not isinstance(ground_truth, dict):
        return 0, len(METADATA_FIELDS)
    correct = 0
    for field in METADATA_FIELDS:
        pred_val = str(prediction.get(field, "")).strip()
        gt_val = str(ground_truth.get(field, "")).strip()
        if field == "company_name":
            if pred_val.lower() == gt_val.lower():
                correct += 1
        elif pred_val == gt_val:
            correct += 1
    return correct, len(METADATA_FIELDS)


def _financial_score(prediction: dict, ground_truth: dict) -> tuple[int, int]:
    if not isinstance(prediction, dict) or not isinstance(ground_truth, dict):
        return 0, len(FINANCIAL_FIELDS)
    correct = 0
    pred_fin = prediction.get("financials", {})
    gt_fin = ground_truth.get("financials", {})
    for field in FINANCIAL_FIELDS:
        pred_val = pred_fin.get(field)
        gt_val = gt_fin.get(field)
        if pred_val is None and gt_val is None:
            correct += 1
            continue
        if pred_val is None or gt_val is None:
            continue
        if field.startswith("eps_"):
            pred_num = _coerce_float(pred_val)
            gt_num = _coerce_float(gt_val)
            if pred_num is not None and gt_num is not None and round(pred_num, 2) == round(gt_num, 2):
                correct += 1
            continue
        pred_num = _coerce_int(pred_val)
        gt_num = _coerce_int(gt_val)
        if pred_num is None or gt_num is None:
            continue
        if gt_num == 0:
            if pred_num == 0:
                correct += 1
            continue
        relative_error = abs(pred_num - gt_num) / abs(gt_num)
        if relative_error <= 0.005:
            correct += 1
    return correct, len(FINANCIAL_FIELDS)


def _cross_validation_ok(prediction: dict) -> bool:
    if not isinstance(prediction, dict):
        return False
    financials = prediction.get("financials", {})
    if not isinstance(financials, dict):
        return False
    assets = financials.get("total_assets")
    liabilities = financials.get("total_liabilities")
    equity = financials.get("stockholders_equity")
    if any(value is None for value in [assets, liabilities, equity]):
        return False
    assets_num = _coerce_int(assets)
    liabilities_num = _coerce_int(liabilities)
    equity_num = _coerce_int(equity)
    if any(value is None for value in [assets_num, liabilities_num, equity_num]):
        return False
    if assets_num == 0 and liabilities_num + equity_num == 0:
        return True
    if assets_num == 0:
        return False
    return abs(assets_num - liabilities_num - equity_num) / abs(assets_num) <= 0.001


def _kendall_tau(list_a: list[str], list_b: list[str]) -> float:
    if set(list_a) != set(list_b):
        common = set(list_a) & set(list_b)
        if len(common) < 2:
            return 0.0
        list_a = [item for item in list_a if item in common]
        list_b = [item for item in list_b if item in common]
    n = len(list_a)
    if n < 2:
        return 1.0 if list_a == list_b else 0.0
    pos_b = {item: index for index, item in enumerate(list_b)}
    mapped = [pos_b[item] for item in list_a]
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if mapped[i] < mapped[j]:
                concordant += 1
            else:
                discordant += 1
    total_pairs = n * (n - 1) / 2
    tau = (concordant - discordant) / total_pairs
    return (tau + 1) / 2


def _qa_score(prediction_path: Path, ground_truth_path: Path) -> float:
    if not prediction_path.exists() or not ground_truth_path.exists():
        return 0.0
    try:
        predictions = _load_json(prediction_path)
        ground_truth = _load_json(ground_truth_path)
    except Exception:
        return 0.0
    if not isinstance(predictions, dict) or not isinstance(ground_truth, dict):
        return 0.0

    scores: list[float] = []
    q1 = predictions.get("Q1")
    gt1 = ground_truth.get("Q1")
    if isinstance(q1, dict) and isinstance(gt1, dict):
        ticker_ok = str(q1.get("ticker", "")).upper() == str(gt1.get("ticker", "")).upper()
        pred_pct = _coerce_float(q1.get("pct_increase"))
        gt_pct = _coerce_float(gt1.get("pct_increase"))
        pct_ok = pred_pct is not None and gt_pct is not None and abs(pred_pct - gt_pct) <= 0.15
        scores.append(1.0 if ticker_ok and pct_ok else 0.0)
    else:
        scores.append(0.0)

    q2 = predictions.get("Q2")
    gt2 = ground_truth.get("Q2")
    if isinstance(q2, dict) and isinstance(gt2, dict):
        ticker_ok = str(q2.get("ticker", "")).upper() == str(gt2.get("ticker", "")).upper()
        pred_year = _coerce_int(q2.get("fiscal_year"))
        gt_year = _coerce_int(gt2.get("fiscal_year"))
        pred_margin = _coerce_float(q2.get("operating_margin_pct"))
        gt_margin = _coerce_float(gt2.get("operating_margin_pct"))
        year_ok = pred_year is not None and gt_year is not None and pred_year == gt_year
        margin_ok = pred_margin is not None and gt_margin is not None and abs(pred_margin - gt_margin) <= 0.05
        scores.append(1.0 if ticker_ok and year_ok and margin_ok else 0.0)
    else:
        scores.append(0.0)

    q3 = predictions.get("Q3")
    gt3 = ground_truth.get("Q3")
    if isinstance(q3, dict) and isinstance(gt3, dict):
        pred_ranking = [str(item).upper() for item in q3.get("ranking", [])]
        gt_ranking = [str(item).upper() for item in gt3.get("ranking", [])]
        scores.append(_kendall_tau(pred_ranking, gt_ranking))
    else:
        scores.append(0.0)
    return sum(scores) / len(scores)


def _determinism_score(
    first_pass_dir: Path,
    run2_dir: Path,
    validation_manifest_path: Path,
) -> float:
    if not run2_dir.exists() or not validation_manifest_path.exists():
        return 0.0
    try:
        manifest = _load_json(validation_manifest_path)
    except Exception:
        return 0.0
    if not isinstance(manifest, dict):
        return 0.0
    filings = manifest.get("validation_filing_stems", [])
    if not isinstance(filings, list) or not filings:
        return 0.0

    matches = 0
    checked = 0
    for stem in filings:
        first_path = first_pass_dir / f"{stem}.json"
        second_path = run2_dir / f"{stem}.json"
        if not first_path.exists() or not second_path.exists():
            checked += 1
            continue
        checked += 1
        try:
            first_payload = _load_json(first_path)
            second_payload = _load_json(second_path)
        except Exception:
            continue
        if isinstance(first_payload, dict) and isinstance(second_payload, dict) and first_payload == second_payload:
            matches += 1
    return matches / checked if checked else 0.0


def score_outputs(
    predictions_dir: Path,
    ground_truth_dir: Path,
    raw_extractions_dir: Path,
    baselines_dir: Path,
    qa_predictions_path: Path,
    qa_ground_truth_path: Path,
    run2_extractions_dir: Path,
    validation_manifest_path: Path,
    cross_validation_expectations_path: Path,
) -> dict:
    gt_files = sorted(ground_truth_dir.glob("*.json"))
    if not gt_files:
        raise FileNotFoundError(f"No ground truth files found in {ground_truth_dir}")

    found_predictions = 0
    schema_valid = 0
    metadata_correct = 0
    metadata_total = 0
    financial_correct = 0
    financial_total = 0
    cross_validation_ok = 0
    anls_scores: list[float] = []
    anls_failed: set[str] = set()
    cross_validation_expectations = _load_json(cross_validation_expectations_path)
    if not isinstance(cross_validation_expectations, dict):
        cross_validation_expectations = {}

    for gt_file in gt_files:
        pred_file = predictions_dir / gt_file.name
        raw_file = raw_extractions_dir / f"{gt_file.stem}.txt"
        baseline_file = baselines_dir / f"{gt_file.stem}.txt"
        anls = 0.0
        if raw_file.exists() and baseline_file.exists():
            anls = compute_anls(
                raw_file.read_text(encoding="utf-8", errors="replace"),
                baseline_file.read_text(encoding="utf-8", errors="replace"),
            )
            if anls < ANLS_THRESHOLD:
                anls_failed.add(gt_file.stem)
            anls_scores.append(anls)
        else:
            anls_scores.append(0.0)
            anls_failed.add(gt_file.stem)

        if not pred_file.exists():
            metadata_total += len(METADATA_FIELDS)
            financial_total += len(FINANCIAL_FIELDS)
            continue

        found_predictions += 1
        try:
            prediction = _load_json(pred_file)
        except Exception:
            metadata_total += len(METADATA_FIELDS)
            financial_total += len(FINANCIAL_FIELDS)
            continue

        ground_truth = _load_json(gt_file)
        if not isinstance(ground_truth, dict):
            raise ValueError(f"ground truth file is not a JSON object: {gt_file}")

        zeroed = gt_file.stem in anls_failed

        if _schema_valid(prediction) and not zeroed:
            schema_valid += 1

        current_meta_correct, current_meta_total = _metadata_score(prediction, ground_truth)
        metadata_total += current_meta_total
        if not zeroed:
            metadata_correct += current_meta_correct

        current_fin_correct, current_fin_total = _financial_score(prediction, ground_truth)
        financial_total += current_fin_total
        if not zeroed:
            financial_correct += current_fin_correct

        expected_cv = bool(cross_validation_expectations.get(gt_file.stem, {}).get("check_passed", False))
        if _cross_validation_ok(prediction) == expected_cv and not zeroed:
            cross_validation_ok += 1

    total_gt = len(gt_files)
    schema_score = schema_valid / total_gt
    metadata_score = metadata_correct / metadata_total if metadata_total else 0.0
    financial_score = financial_correct / financial_total if financial_total else 0.0
    anls_score = sum(anls_scores) / len(anls_scores) if anls_scores else 0.0
    cross_validation_score = cross_validation_ok / total_gt
    completeness_score = found_predictions / total_gt
    qa_score = _qa_score(qa_predictions_path, qa_ground_truth_path)
    determinism_score = _determinism_score(predictions_dir, run2_extractions_dir, validation_manifest_path)

    aggregate = (
        WEIGHTS["schema_compliance"] * schema_score
        + WEIGHTS["metadata_accuracy"] * metadata_score
        + WEIGHTS["financial_accuracy"] * financial_score
        + WEIGHTS["anls_verification"] * anls_score
        + WEIGHTS["analytical_qa"] * qa_score
        + WEIGHTS["cross_validation"] * cross_validation_score
        + WEIGHTS["completeness"] * completeness_score
        + WEIGHTS["determinism"] * determinism_score
    )

    return {
        "score": round(aggregate, 4),
        "passing_threshold": 0.75,
        "component_scores": {
            "schema_compliance": round(schema_score, 4),
            "metadata_accuracy": round(metadata_score, 4),
            "financial_accuracy": round(financial_score, 4),
            "anls_verification": round(anls_score, 4),
            "analytical_qa": round(qa_score, 4),
            "cross_validation": round(cross_validation_score, 4),
            "completeness": round(completeness_score, 4),
            "determinism": round(determinism_score, 4),
        },
        "counts": {
            "ground_truth_files": total_gt,
            "prediction_files_found": found_predictions,
            "schema_valid": schema_valid,
            "anls_failures": len(anls_failed),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--raw-extractions", required=True)
    parser.add_argument("--baselines", required=True)
    parser.add_argument("--qa-predictions", required=True)
    parser.add_argument("--qa-ground-truth", required=True)
    parser.add_argument("--run2-extractions", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--cross-validation-expectations", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = score_outputs(
        predictions_dir=Path(args.predictions),
        ground_truth_dir=Path(args.ground_truth),
        raw_extractions_dir=Path(args.raw_extractions),
        baselines_dir=Path(args.baselines),
        qa_predictions_path=Path(args.qa_predictions),
        qa_ground_truth_path=Path(args.qa_ground_truth),
        run2_extractions_dir=Path(args.run2_extractions),
        validation_manifest_path=Path(args.validation_manifest),
        cross_validation_expectations_path=Path(args.cross_validation_expectations),
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
