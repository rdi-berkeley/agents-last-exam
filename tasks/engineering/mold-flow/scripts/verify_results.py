"""
verify_results.py — Primary evaluation script (runs on remote Windows VM)

Compares agent's results.json against reference results.json using field-by-field
numerical comparison with configurable relative tolerance.

Requirements:
    Python 3.x (no extra packages needed)

Usage:
    python verify_results.py --agent PATH_TO_AGENT_JSON --ref PATH_TO_REF_JSON

Output:
    JSON to stdout, e.g.:
    {"score": 0.85, "matched_fields": 17, "total_fields": 20, "field_details": {...}}

Exit codes:
    0 = success (score printed to stdout as JSON)
    1 = error (missing files, parse failure, etc.)
"""

import sys
import json
import argparse
import os


# ---------------------------------------------------------------------------
# Tolerance for numerical comparison
# ---------------------------------------------------------------------------
RELATIVE_TOLERANCE = 0.01  # 1% relative tolerance


def flatten_json(obj, parent_key="", sep="."):
    """Flatten a nested JSON dict into dot-separated key-value pairs.

    Example: {"a": {"b": 1}} → {"a.b": 1}
    """
    items = []
    for k, v in obj.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_json(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def compare_values(agent_val, ref_val, tolerance=RELATIVE_TOLERANCE):
    """Compare two values with relative tolerance.

    Returns:
        (match: bool, detail: str)
    """
    if agent_val is None:
        return False, "null (not filled)"

    if ref_val is None:
        return True, "ref is null (skipped)"

    # Both should be numeric
    try:
        a = float(agent_val)
        r = float(ref_val)
    except (TypeError, ValueError):
        # String comparison for non-numeric fields
        return str(agent_val) == str(ref_val), f"string compare: {agent_val} vs {ref_val}"

    if r == 0:
        # Avoid division by zero; use absolute comparison
        match = abs(a) < 1e-6
        return match, f"agent={a}, ref={r}, abs_diff={abs(a - r)}"

    rel_error = abs(a - r) / abs(r)
    match = rel_error <= tolerance
    return match, f"agent={a:.6g}, ref={r:.6g}, rel_error={rel_error:.4%}"


def verify(agent_path, ref_path):
    """Compare agent results vs reference and return scoring dict."""

    # Load files
    with open(agent_path, "r", encoding="utf-8") as f:
        agent_data = json.load(f)

    with open(ref_path, "r", encoding="utf-8") as f:
        ref_data = json.load(f)

    # Flatten both
    agent_flat = flatten_json(agent_data)
    ref_flat = flatten_json(ref_data)

    # Compare field by field (use reference keys as the ground truth)
    field_details = {}
    matched = 0
    total = 0

    for key, ref_val in ref_flat.items():
        if ref_val is None:
            # Skip fields that are null in reference (shouldn't happen but be safe)
            continue

        total += 1
        agent_val = agent_flat.get(key, None)
        match, detail = compare_values(agent_val, ref_val)
        field_details[key] = {"match": match, "detail": detail}
        if match:
            matched += 1

    score = matched / total if total > 0 else 0.0

    return {
        "score": round(score, 4),
        "matched_fields": matched,
        "total_fields": total,
        "tolerance": RELATIVE_TOLERANCE,
        "field_details": field_details,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare agent results.json vs reference results.json"
    )
    parser.add_argument("--agent", required=True, help="Path to agent results.json")
    parser.add_argument("--ref", required=True, help="Path to reference results.json")
    args = parser.parse_args()

    for path, label in [(args.agent, "agent results"), (args.ref, "reference results")]:
        if not os.path.exists(path):
            result = {"score": 0.0, "error": f"{label} not found: {path}"}
            print(json.dumps(result))
            sys.exit(1)

    try:
        result = verify(args.agent, args.ref)
        print(json.dumps(result))
        sys.exit(0)
    except Exception as e:
        result = {"score": 0.0, "error": str(e)}
        print(json.dumps(result))
        sys.exit(1)


if __name__ == "__main__":
    main()
