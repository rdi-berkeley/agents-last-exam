"""Local scorer for eeg_psd_feature_extraction."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass

SUMMARY_FIELDS = [
    "subject_id",
    "sampling_rate_hz",
    "welch_window_seconds",
    "welch_window_samples",
    "alpha_global_abs_power",
    "alpha_o1_abs_power",
    "alpha_o2_abs_power",
    "theta_alpha_ratio",
]
CHANNEL_FIELDS = ["channel", "alpha_abs_power", "alpha_rank_desc"]
NUMERIC_KEYS = [
    "alpha_global_abs_power",
    "alpha_o1_abs_power",
    "alpha_o2_abs_power",
    "theta_alpha_ratio",
]


@dataclass(frozen=True)
class ScoreResult:
    score: float
    passed: bool
    reason: str
    hard_gate: str | None
    component_scores: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


def _read_csv_dicts(text: str) -> tuple[list[str], list[dict[str, str]]]:
    reader = csv.DictReader(io.StringIO(text))
    return reader.fieldnames or [], list(reader)


def _is_png(payload: bytes) -> bool:
    return isinstance(payload, (bytes, bytearray)) and len(payload) > 100 and bytes(payload).startswith(b"\x89PNG\r\n\x1a\n")


def _relative_component_score(agent_value: float, reference_value: float) -> float:
    denom = abs(reference_value) if abs(reference_value) > 1e-15 else 1.0
    rel_err = abs(agent_value - reference_value) / denom
    if rel_err <= 0.05:
        return 1.0
    if rel_err <= 0.15:
        return 0.5
    return 0.0


def score_output_bundle(
    *,
    summary_csv: str,
    channel_csv: str,
    psd_plot_bytes: bytes,
    topomap_bytes: bytes,
    reference_summary_csv: str,
    reference_channel_csv: str,
    expected_subject_id: str,
) -> ScoreResult:
    summary_fields, summary_rows = _read_csv_dicts(summary_csv)
    ref_summary_fields, ref_summary_rows = _read_csv_dicts(reference_summary_csv)
    channel_fields, channel_rows = _read_csv_dicts(channel_csv)
    ref_channel_fields, ref_channel_rows = _read_csv_dicts(reference_channel_csv)

    if summary_fields != SUMMARY_FIELDS or len(summary_rows) != 1:
        return ScoreResult(0.0, False, "hard_gate_failure", "summary_schema_or_row_count", {})
    if ref_summary_fields != SUMMARY_FIELDS or len(ref_summary_rows) != 1:
        return ScoreResult(0.0, False, "reference_error", "reference_summary_schema", {})
    if channel_fields != CHANNEL_FIELDS or not channel_rows:
        return ScoreResult(0.0, False, "hard_gate_failure", "channel_schema_or_empty", {})
    if ref_channel_fields != CHANNEL_FIELDS or not ref_channel_rows:
        return ScoreResult(0.0, False, "reference_error", "reference_channel_schema", {})

    summary_row = summary_rows[0]
    ref_summary_row = ref_summary_rows[0]

    if summary_row["subject_id"] != expected_subject_id:
        return ScoreResult(0.0, False, "hard_gate_failure", "subject_id_mismatch", {})

    channel_names = [row["channel"] for row in channel_rows]
    if len(set(channel_names)) != len(channel_names):
        return ScoreResult(0.0, False, "hard_gate_failure", "duplicate_channel_names", {})
    if "O1" not in channel_names or "O2" not in channel_names:
        return ScoreResult(0.0, False, "hard_gate_failure", "missing_o1_or_o2", {})

    if not _is_png(psd_plot_bytes):
        return ScoreResult(0.0, False, "hard_gate_failure", "invalid_psd_png", {})
    if not _is_png(topomap_bytes):
        return ScoreResult(0.0, False, "hard_gate_failure", "invalid_topomap_png", {})

    component_scores: dict[str, float] = {}
    try:
        for key in NUMERIC_KEYS:
            component_scores[key] = _relative_component_score(float(summary_row[key]), float(ref_summary_row[key]))
    except Exception:
        return ScoreResult(0.0, False, "hard_gate_failure", "numeric_parse_failure", {})

    try:
        rank_by_channel = {row["channel"]: int(row["alpha_rank_desc"]) for row in channel_rows}
        ref_rank_by_channel = {row["channel"]: int(row["alpha_rank_desc"]) for row in ref_channel_rows}
    except Exception:
        return ScoreResult(0.0, False, "hard_gate_failure", "channel_rank_parse_failure", {})

    # Keep the original "top 3" expectation when the hidden subject really has
    # dominant occipital alpha, but do not penalize exact-reference outputs for
    # subjects whose best occipital rank is slightly lower.
    reference_best_occipital_rank = min(ref_rank_by_channel["O1"], ref_rank_by_channel["O2"])
    allowed_best_occipital_rank = max(3, reference_best_occipital_rank)
    spatial = 1.0 if min(rank_by_channel["O1"], rank_by_channel["O2"]) <= allowed_best_occipital_rank else 0.0
    component_scores["spatial_consistency"] = spatial

    score = sum(component_scores.values()) / 5.0
    passed = score >= 1.0
    reason = "ok" if passed else "below_perfect_threshold"
    return ScoreResult(score=score, passed=passed, reason=reason, hard_gate=None, component_scores=component_scores)


def main() -> int:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--channel", required=True)
    parser.add_argument("--psd", required=True)
    parser.add_argument("--topomap", required=True)
    parser.add_argument("--ref-summary", required=True)
    parser.add_argument("--ref-channel", required=True)
    parser.add_argument("--subject-id", required=True)
    args = parser.parse_args()

    result = score_output_bundle(
        summary_csv=Path(args.summary).read_text(encoding="utf-8"),
        channel_csv=Path(args.channel).read_text(encoding="utf-8"),
        psd_plot_bytes=Path(args.psd).read_bytes(),
        topomap_bytes=Path(args.topomap).read_bytes(),
        reference_summary_csv=Path(args.ref_summary).read_text(encoding="utf-8"),
        reference_channel_csv=Path(args.ref_channel).read_text(encoding="utf-8"),
        expected_subject_id=args.subject_id,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
