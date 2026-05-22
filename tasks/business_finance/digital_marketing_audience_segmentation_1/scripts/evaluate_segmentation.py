"""Scoring logic for digital_marketing_audience_segmentation_1.

Compares agent output (segment_definition.json, audience_roster.csv,
overlap_report.tsv) against reference files and returns a weighted score.

Hard gates (score=0):
  - Any output file missing or unparsable
  - PII fields (age, gender, city, state) present in roster columns
  - Audience size below governance minimum (50) when reference is larger
"""

import csv
import io
import json
import logging

logger = logging.getLogger(__name__)

PII_FIELDS = {"age", "gender", "city", "state"}
MIN_AUDIENCE_SIZE = 50

# Scoring weights
W_SEG_STRUCTURE = 0.15
W_AUDIENCE_STATS = 0.10
W_ROSTER_IDS = 0.35
W_ROSTER_ELIG = 0.20
W_OVERLAP_COUNTS = 0.15
W_OVERLAP_PCT = 0.05


def score_segmentation(
    seg_out: str,
    roster_out: str,
    overlap_out: str,
    seg_ref: str,
    roster_ref: str,
    overlap_ref: str,
) -> dict:
    """Return {"score": float, "details": dict} comparing output vs reference."""
    details = {}

    # --- Parse all files (hard gate) ---
    try:
        seg_out_obj = json.loads(seg_out)
    except (json.JSONDecodeError, ValueError):
        return {"score": 0.0, "details": {"error": "segment_definition.json unparsable"}}
    try:
        seg_ref_obj = json.loads(seg_ref)
    except (json.JSONDecodeError, ValueError):
        return {"score": 0.0, "details": {"error": "ref segment_definition.json unparsable"}}

    try:
        roster_out_rows = _parse_csv(roster_out)
    except Exception:
        return {"score": 0.0, "details": {"error": "audience_roster.csv unparsable"}}
    try:
        roster_ref_rows = _parse_csv(roster_ref)
    except Exception:
        return {"score": 0.0, "details": {"error": "ref audience_roster.csv unparsable"}}

    try:
        overlap_out_rows = _parse_tsv(overlap_out)
    except Exception:
        return {"score": 0.0, "details": {"error": "overlap_report.tsv unparsable"}}
    try:
        overlap_ref_rows = _parse_tsv(overlap_ref)
    except Exception:
        return {"score": 0.0, "details": {"error": "ref overlap_report.tsv unparsable"}}

    # --- Hard gate: PII fields in output roster ---
    out_cols = set()
    if roster_out_rows:
        out_cols = set(roster_out_rows[0].keys())
    else:
        first_line = roster_out.strip().split("\n")[0] if roster_out.strip() else ""
        if first_line:
            out_cols = {c.strip() for c in first_line.split(",")}
    pii_present = PII_FIELDS & {c.strip().lower() for c in out_cols}
    if pii_present:
        return {
            "score": 0.0,
            "details": {"error": f"PII fields present in roster: {sorted(pii_present)}"},
        }

    # --- Hard gate: audience size ---
    ref_total = seg_ref_obj.get("audience_stats", {}).get("total_qualifying", 0)
    out_total = len(roster_out_rows)
    if out_total < MIN_AUDIENCE_SIZE and ref_total >= MIN_AUDIENCE_SIZE:
        return {
            "score": 0.0,
            "details": {"error": f"audience size {out_total} below minimum {MIN_AUDIENCE_SIZE}"},
        }

    # --- Score components ---
    seg_score = _score_segment_definition(seg_out_obj, seg_ref_obj)
    details["segment_definition"] = seg_score

    stats_score = _score_audience_stats(seg_out_obj, seg_ref_obj)
    details["audience_stats"] = stats_score

    id_score = _score_roster_ids(roster_out_rows, roster_ref_rows)
    details["roster_ids"] = id_score

    elig_score = _score_roster_eligibility(roster_out_rows, roster_ref_rows)
    details["roster_eligibility"] = elig_score

    ov_count_score = _score_overlap_counts(overlap_out_rows, overlap_ref_rows)
    details["overlap_counts"] = ov_count_score

    ov_pct_score = _score_overlap_pct(overlap_out_rows, overlap_ref_rows)
    details["overlap_pct"] = ov_pct_score

    final = (
        W_SEG_STRUCTURE * seg_score
        + W_AUDIENCE_STATS * stats_score
        + W_ROSTER_IDS * id_score
        + W_ROSTER_ELIG * elig_score
        + W_OVERLAP_COUNTS * ov_count_score
        + W_OVERLAP_PCT * ov_pct_score
    )
    return {"score": round(final, 4), "details": details}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_csv(text: str) -> list:
    reader = csv.DictReader(io.StringIO(text))
    return list(reader)


def _parse_tsv(text: str) -> list:
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    return list(reader)


def _predicate_key(p: dict) -> tuple:
    """Normalize a filter-predicate dict to a comparable tuple."""
    return (str(p.get("field", "")), str(p.get("operator", "")), str(p.get("value", "")))


def _normalize_bool(val: str) -> str:
    v = val.lower().strip()
    if v in ("1", "true", "yes"):
        return "1"
    if v in ("0", "false", "no", ""):
        return "0"
    return v


def _score_segment_definition(out: dict, ref: dict) -> float:
    checks = 0
    total = 0

    # filter_predicates set-equivalence
    total += 1
    out_preds = {_predicate_key(p) for p in out.get("filter_predicates", [])}
    ref_preds = {_predicate_key(p) for p in ref.get("filter_predicates", [])}
    if out_preds == ref_preds:
        checks += 1

    # suppression_rules (field/operator/value match)
    total += 1
    out_supp = {_predicate_key(s) for s in out.get("suppression_rules", [])}
    ref_supp = {_predicate_key(s) for s in ref.get("suppression_rules", [])}
    if out_supp == ref_supp:
        checks += 1

    # activation_channels keys
    total += 1
    out_ch = set(out.get("activation_channels", {}).keys())
    ref_ch = set(ref.get("activation_channels", {}).keys())
    if out_ch == ref_ch:
        checks += 1

    # governance_applied.pii_fields_removed (accept superset, e.g. includes state)
    total += 1
    out_pii = set(out.get("governance_applied", {}).get("pii_fields_removed", []))
    ref_pii = set(ref.get("governance_applied", {}).get("pii_fields_removed", []))
    if out_pii == ref_pii or out_pii >= ref_pii:
        checks += 1

    return checks / total if total > 0 else 0.0


def _score_audience_stats(out: dict, ref: dict) -> float:
    out_stats = out.get("audience_stats", {})
    ref_stats = ref.get("audience_stats", {})
    fields = ["total_qualifying", "sms_eligible", "push_eligible", "any_channel_eligible"]
    checks = 0
    total = len(fields)
    for f in fields:
        out_val = out_stats.get(f)
        ref_val = ref_stats.get(f, 0)
        if out_val is None:
            continue
        try:
            out_val = float(out_val)
            ref_val = float(ref_val)
        except (ValueError, TypeError):
            continue
        if ref_val == 0:
            if out_val == 0:
                checks += 1
        elif abs(out_val - ref_val) / abs(ref_val) <= 0.02:
            checks += 1
    return checks / total if total > 0 else 0.0


def _score_roster_ids(out_rows: list, ref_rows: list) -> float:
    out_ids = {r.get("customer_id", "").strip() for r in out_rows} - {""}
    ref_ids = {r.get("customer_id", "").strip() for r in ref_rows} - {""}
    if not ref_ids:
        return 1.0 if not out_ids else 0.0
    union = out_ids | ref_ids
    if not union:
        return 1.0
    return len(out_ids & ref_ids) / len(union)


def _score_roster_eligibility(out_rows: list, ref_rows: list) -> float:
    ref_by_id = {r.get("customer_id", "").strip(): r for r in ref_rows}
    elig_fields = ["sms_eligible", "push_eligible", "any_channel_eligible"]
    matched = 0
    total = 0
    for out_row in out_rows:
        cid = out_row.get("customer_id", "").strip()
        ref_row = ref_by_id.get(cid)
        if ref_row is None:
            continue
        for f in elig_fields:
            total += 1
            if _normalize_bool(str(out_row.get(f, ""))) == _normalize_bool(
                str(ref_row.get(f, ""))
            ):
                matched += 1
    return matched / total if total > 0 else 0.0


def _score_overlap_counts(out_rows: list, ref_rows: list) -> float:
    ref_by_id = {r.get("existing_audience_id", "").strip(): r for r in ref_rows}
    if not ref_by_id:
        return 1.0 if not out_rows else 0.0
    out_by_id = {r.get("existing_audience_id", "").strip(): r for r in out_rows}
    matched = 0
    total = len(ref_by_id)
    for aid, ref_row in ref_by_id.items():
        out_row = out_by_id.get(aid)
        if out_row is None:
            continue
        try:
            if int(out_row.get("overlap_count", -1)) == int(ref_row.get("overlap_count", 0)):
                matched += 1
        except (ValueError, TypeError):
            pass
    return matched / total


def _score_overlap_pct(out_rows: list, ref_rows: list) -> float:
    ref_by_id = {r.get("existing_audience_id", "").strip(): r for r in ref_rows}
    if not ref_by_id:
        return 1.0 if not out_rows else 0.0
    out_by_id = {r.get("existing_audience_id", "").strip(): r for r in out_rows}
    matched = 0
    total = len(ref_by_id)
    for aid, ref_row in ref_by_id.items():
        out_row = out_by_id.get(aid)
        if out_row is None:
            continue
        try:
            out_pct = float(out_row.get("overlap_pct", -999))
            ref_pct = float(ref_row.get("overlap_pct", 0))
            if abs(out_pct - ref_pct) <= 0.5:
                matched += 1
        except (ValueError, TypeError):
            pass
    return matched / total
