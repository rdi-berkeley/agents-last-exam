"""Verify NSCLC radiomics Cox-signature outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

PASS_THRESHOLD = 0.55


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _patient_id(row: dict[str, str]) -> str:
    for key in ("PatientID", "patient_id", "PatientId", "id"):
        if key in row and row[key]:
            return row[key].strip()
    raise ValueError("missing patient id column")


def _harrell_c_index(durations: list[float], events: list[int], risks: list[float]) -> float:
    concordant = 0.0
    comparable = 0.0
    n = len(durations)
    for i in range(n):
        for j in range(i + 1, n):
            if durations[i] == durations[j]:
                continue
            if durations[i] < durations[j]:
                early, late = i, j
            else:
                early, late = j, i
            if events[early] == 0:
                continue
            comparable += 1.0
            if risks[early] > risks[late]:
                concordant += 1.0
            elif risks[early] == risks[late]:
                concordant += 0.5
    if comparable == 0:
        return 0.5
    return concordant / comparable


def _fail(reason: str) -> dict:
    return {"score": 0.0, "passed": False, "reason": reason}


def verify(output_dir: Path, reference_dir: Path) -> dict:
    risk_path = output_dir / "risk_scores.csv"
    truth_path = reference_dir / "ground_truth.csv"

    for path in (risk_path, truth_path):
        if not path.exists():
            return _fail(f"missing required file: {path}")

    try:
        risks = _read_csv(risk_path)
        truth = _read_csv(truth_path)
    except Exception as exc:
        return _fail(f"failed to load inputs: {exc}")

    truth_by_pid = {}
    for row in truth:
        try:
            pid = _patient_id(row)
            truth_by_pid[pid] = {
                "time": float(row["survival_time_days"]),
                "event": int(float(row["deadstatus.event"])),
            }
        except Exception as exc:
            return _fail(f"invalid ground truth row: {exc}")

    if len(risks) != len(truth_by_pid):
        return _fail(f"risk_scores.csv has {len(risks)} rows, expected {len(truth_by_pid)}")

    seen: set[str] = set()
    scored_rows: list[dict] = []
    for row in risks:
        try:
            pid = _patient_id(row)
            score = float(row["risk_score"])
        except Exception as exc:
            return _fail(f"invalid risk_scores row: {exc}")
        if not math.isfinite(score):
            return _fail(f"non-finite risk_score for {pid}")
        if pid in seen:
            return _fail(f"duplicate patient id: {pid}")
        if pid not in truth_by_pid:
            return _fail(f"unknown patient id in risk_scores.csv: {pid}")
        seen.add(pid)
        scored_rows.append({"pid": pid, "risk": score, **truth_by_pid[pid]})

    if set(truth_by_pid) != seen:
        missing = sorted(set(truth_by_pid) - seen)[:5]
        return _fail(f"missing patients in risk_scores.csv: {missing}")

    c_index = _harrell_c_index(
        [r["time"] for r in scored_rows],
        [r["event"] for r in scored_rows],
        [r["risk"] for r in scored_rows],
    )
    if c_index == 0.5:
        return _fail("held-out C-index is exactly 0.5")

    return {
        "score": round(c_index, 6) if c_index >= PASS_THRESHOLD else 0.0,
        "passed": c_index >= PASS_THRESHOLD,
        "c_index": round(c_index, 6),
        "pass_threshold": PASS_THRESHOLD,
        "n_patients": len(seen),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    result = verify(Path(args.output_dir), Path(args.reference_dir))
    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
