"""Generate form1040_gt_alt.json for each taxform variant.

For each variant, the alt reference uses the OTHER standard deduction value
(IRS correct vs form sidebar) and recomputes all cascading fields.

Usage:
    python generate_alt_references.py [--dry-run] [--task taxform_1_1]
"""

import argparse
import json
import subprocess
import copy
from pathlib import Path

CORRECT_STD_DED = {"single": 15000, "mfj": 30000, "hoh": 22500}
SIDEBAR_STD_DED = {"single": 15750, "mfj": 31500, "hoh": 23625}

STD_DED_TO_STATUS = {}
for status in CORRECT_STD_DED:
    STD_DED_TO_STATUS[CORRECT_STD_DED[status]] = status
    STD_DED_TO_STATUS[SIDEBAR_STD_DED[status]] = status

TAX_BRACKETS = {
    "single": [
        (11925, 0.10), (48475, 0.12), (103350, 0.22),
        (197300, 0.24), (250525, 0.32), (626350, 0.35), (float("inf"), 0.37),
    ],
    "mfj": [
        (23850, 0.10), (96950, 0.12), (206700, 0.22),
        (394600, 0.24), (501050, 0.32), (751600, 0.35), (float("inf"), 0.37),
    ],
    "hoh": [
        (17000, 0.10), (64850, 0.12), (103350, 0.22),
        (197300, 0.24), (250500, 0.32), (626350, 0.35), (float("inf"), 0.37),
    ],
}

TASKS = [f"taxform_{i}_1" for i in range(1, 7)]
VARIANTS = [f"variant_{i}" for i in range(1, 6)]
GCS_PREFIX = "gs://ale-data-all/finance"


def compute_tax(taxable_income: float, filing_status: str) -> float:
    brackets = TAX_BRACKETS[filing_status]
    tax = 0.0
    prev = 0.0
    for limit, rate in brackets:
        if taxable_income <= prev:
            break
        taxed = min(taxable_income, limit) - prev
        tax += taxed * rate
        prev = limit
    return round(tax, 2)


def parse_num(val: str) -> float:
    try:
        return float(val.replace(",", "").strip())
    except (ValueError, AttributeError):
        return 0.0


def fmt_num(val: float) -> str:
    if val == int(val):
        return f"{int(val):,}"
    return f"{val:,.2f}"


def gsutil_cat(path: str) -> str:
    result = subprocess.run(
        ["gsutil", "cat", path], capture_output=True, text=True, timeout=30
    )
    if result.returncode != 0:
        raise FileNotFoundError(f"gsutil cat failed for {path}: {result.stderr.strip()}")
    return result.stdout


def gsutil_cp(local_path: str, gcs_path: str):
    subprocess.run(
        ["gsutil", "cp", local_path, gcs_path], check=True, capture_output=True, timeout=30
    )


def get_field_val(fields: dict, name: str) -> float:
    entry = fields.get(name, {})
    return parse_num(entry.get("value", "0"))


def set_field_val(fields: dict, name: str, val: float):
    if name in fields:
        fields[name]["value"] = fmt_num(val)


def generate_alt_for_form(primary: dict, form_name: str = "form1040") -> dict | None:
    fields = primary.get("fields", {})
    primary_12e = get_field_val(fields, "line_12e")

    if primary_12e not in STD_DED_TO_STATUS:
        print(f"    WARNING: line_12e={primary_12e} not recognized, skipping")
        return None

    filing_status = STD_DED_TO_STATUS[primary_12e]
    correct = CORRECT_STD_DED[filing_status]
    sidebar = SIDEBAR_STD_DED[filing_status]
    alt_12e = sidebar if primary_12e == correct else correct

    alt = copy.deepcopy(primary)
    af = alt["fields"]

    set_field_val(af, "line_12e", alt_12e)

    line_13a = get_field_val(fields, "line_13a")
    line_13b = get_field_val(fields, "line_13b")
    alt_14 = alt_12e + line_13a + line_13b
    set_field_val(af, "line_14", alt_14)

    line_11a = get_field_val(fields, "line_11a")
    if "line_11b" in fields:
        line_11_for_calc = get_field_val(fields, "line_11b")
        if line_11_for_calc > 0:
            line_11a = line_11_for_calc

    alt_15 = max(0.0, line_11a - alt_14)
    set_field_val(af, "line_15", alt_15)

    alt_16 = compute_tax(alt_15, filing_status)
    set_field_val(af, "line_16", alt_16)

    line_17 = get_field_val(fields, "line_17")
    alt_18 = alt_16 + line_17
    set_field_val(af, "line_18", alt_18)

    line_19 = get_field_val(fields, "line_19")
    line_20 = get_field_val(fields, "line_20")
    line_21 = get_field_val(fields, "line_21")
    alt_22 = max(0.0, alt_18 - line_19 - line_20 - line_21)
    set_field_val(af, "line_22", alt_22)

    line_23 = get_field_val(fields, "line_23")
    alt_24 = alt_22 + line_23
    set_field_val(af, "line_24", alt_24)

    line_33 = get_field_val(fields, "line_33")

    if "line_34" in fields:
        alt_34 = max(0.0, line_33 - alt_24)
        set_field_val(af, "line_34", alt_34)
        if "line_35a" in fields:
            set_field_val(af, "line_35a", alt_34)

    if "line_37" in fields:
        alt_37 = max(0.0, alt_24 - line_33)
        set_field_val(af, "line_37", alt_37)

    return alt


def process_variant(task: str, variant: str, dry_run: bool):
    gcs_ref_dir = f"{GCS_PREFIX}/{task}/{variant}/reference"

    for form_name in ["form1040"]:
        gt_path = f"{gcs_ref_dir}/{form_name}_gt.json"
        try:
            raw = gsutil_cat(gt_path)
        except FileNotFoundError:
            print(f"  {variant}: {form_name}_gt.json not found, skipping")
            continue

        primary = json.loads(raw)
        alt = generate_alt_for_form(primary, form_name)
        if alt is None:
            continue

        primary_12e = get_field_val(primary["fields"], "line_12e")
        alt_12e = get_field_val(alt["fields"], "line_12e")
        primary_16 = get_field_val(primary["fields"], "line_16")
        alt_16 = get_field_val(alt["fields"], "line_16")
        print(f"  {variant}: line_12e {fmt_num(primary_12e)} → alt {fmt_num(alt_12e)}, "
              f"line_16 {fmt_num(primary_16)} → alt {fmt_num(alt_16)}")

        if dry_run:
            continue

        alt_path = f"{gcs_ref_dir}/{form_name}_gt_alt.json"
        tmp = Path(f"/tmp/alt_ref_{task}_{variant}_{form_name}.json")
        tmp.write_text(json.dumps(alt, indent=2, ensure_ascii=False))
        gsutil_cp(str(tmp), alt_path)
        tmp.unlink()
        print(f"    → uploaded to {alt_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--task", type=str, help="Process only this task (e.g. taxform_1_1)")
    args = parser.parse_args()

    tasks = [args.task] if args.task else TASKS
    for task in tasks:
        print(f"\n=== {task} ===")
        for variant in VARIANTS:
            process_variant(task, variant, args.dry_run)


if __name__ == "__main__":
    main()
