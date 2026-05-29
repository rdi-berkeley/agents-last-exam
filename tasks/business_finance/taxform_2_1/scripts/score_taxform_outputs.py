import argparse
import json
from pathlib import Path
from typing import Any

ALLOWED_TOP_LEVEL_KEYS = {"fields", "pagination"}
TEXT_LIKE_FIELD_TYPES = {"text", "textarea", "radio", "select"}


def _normalize_numeric(value: str) -> float | None:
    try:
        return float(value.replace(",", "").strip())
    except Exception:
        return None


def _is_informative(field_type: str, value: Any) -> bool:
    if field_type == "checkbox":
        return isinstance(value, list) and len(value) > 0
    if field_type in TEXT_LIKE_FIELD_TYPES:
        return isinstance(value, str) and value.strip() != ""
    return bool(value)


def _is_valid_value_shape(field_type: str, value: Any) -> bool:
    if field_type == "checkbox":
        return isinstance(value, list)
    if field_type in TEXT_LIKE_FIELD_TYPES:
        return isinstance(value, str)
    return False


def _blank_value_for_type(field_type: str) -> Any:
    if field_type == "checkbox":
        return []
    if field_type in TEXT_LIKE_FIELD_TYPES:
        return ""
    return None


def _values_match(field_name: str, field_type: str, expected: Any, actual: Any) -> bool:
    if field_type == "checkbox":
        return set(expected) == set(actual)

    if field_name.startswith("line_") and isinstance(expected, str) and isinstance(actual, str):
        lhs = _normalize_numeric(expected)
        rhs = _normalize_numeric(actual)
        if lhs is not None and rhs is not None:
            return abs(lhs - rhs) < 0.01

    return expected == actual


def _hard_fail(reason: str, **extra: Any) -> dict[str, Any]:
    payload = {
        "score": 0.0,
        "matched_informative_fields": 0,
        "total_informative_fields": 0,
        "hard_fail_reason": reason,
    }
    payload.update(extra)
    return payload


def score_single_form(
    reference_payload: dict[str, Any],
    output_payload: dict[str, Any],
    alt_reference_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if set(output_payload) - ALLOWED_TOP_LEVEL_KEYS:
        return _hard_fail("unexpected_top_level_keys")

    reference_fields = reference_payload.get("fields")
    output_fields = output_payload.get("fields")
    if not isinstance(reference_fields, dict):
        return _hard_fail("invalid_reference_fields")
    if not isinstance(output_fields, dict):
        return _hard_fail("missing_or_invalid_fields")

    alt_fields = (alt_reference_payload or {}).get("fields", {})

    reference_names = set(reference_fields)
    output_names = set(output_fields)
    missing_fields = sorted(reference_names - output_names)
    extra_fields = sorted(output_names - reference_names)

    informative_total = 0
    informative_matched = 0
    mismatched_fields: list[str] = []
    invalid_field_entries: list[str] = []
    field_type_mismatches: list[str] = []
    unexpected_value_shapes: list[str] = []
    missing_informative_fields: list[str] = []

    for field_name, expected_entry in reference_fields.items():
        actual_entry = output_fields.get(field_name)
        if not isinstance(expected_entry, dict):
            return _hard_fail("invalid_reference_field_entry", field_name=field_name)

        expected_type = expected_entry.get("type")
        expected_value = expected_entry.get("value")

        if not _is_informative(expected_type, expected_value):
            continue

        informative_total += 1

        if actual_entry is None:
            missing_informative_fields.append(field_name)
            mismatched_fields.append(field_name)
            continue

        if not isinstance(actual_entry, dict):
            invalid_field_entries.append(field_name)
            mismatched_fields.append(field_name)
            continue

        actual_type = actual_entry.get("type")
        if expected_type != actual_type:
            field_type_mismatches.append(field_name)
            mismatched_fields.append(field_name)
            continue

        actual_value = actual_entry.get("value", _blank_value_for_type(expected_type))
        if not _is_valid_value_shape(expected_type, actual_value):
            unexpected_value_shapes.append(field_name)
            mismatched_fields.append(field_name)
            continue

        if _values_match(field_name, expected_type, expected_value, actual_value):
            informative_matched += 1
        elif field_name in alt_fields:
            alt_entry = alt_fields[field_name]
            alt_value = alt_entry.get("value")
            if _values_match(field_name, expected_type, alt_value, actual_value):
                informative_matched += 1
            else:
                mismatched_fields.append(field_name)
        else:
            mismatched_fields.append(field_name)

    score = informative_matched / informative_total if informative_total else 0.0
    return {
        "score": score,
        "matched_informative_fields": informative_matched,
        "total_informative_fields": informative_total,
        "hard_fail_reason": None,
        "missing_fields": missing_fields,
        "extra_fields": extra_fields,
        "missing_informative_fields": missing_informative_fields,
        "invalid_field_entries": invalid_field_entries,
        "field_type_mismatches": field_type_mismatches,
        "unexpected_value_shapes": unexpected_value_shapes,
        "mismatched_fields": mismatched_fields,
    }


def score_variant_outputs(
    reference_payloads: dict[str, dict[str, Any]],
    output_payloads: dict[str, dict[str, Any]],
    alt_reference_payloads: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    missing_forms = sorted(set(reference_payloads) - set(output_payloads))
    extra_forms = sorted(set(output_payloads) - set(reference_payloads))
    if missing_forms:
        return _hard_fail("missing_output_files", missing_forms=missing_forms)
    if extra_forms:
        return _hard_fail("unexpected_output_files", extra_forms=extra_forms)

    matched_total = 0
    informative_total = 0
    per_form: dict[str, Any] = {}

    for form_name in sorted(reference_payloads):
        alt_payload = (alt_reference_payloads or {}).get(form_name)
        form_result = score_single_form(reference_payloads[form_name], output_payloads[form_name], alt_payload)
        per_form[form_name] = form_result
        if form_result["hard_fail_reason"] is not None:
            return _hard_fail(
                form_result["hard_fail_reason"],
                form_name=form_name,
                per_form=per_form,
            )
        matched_total += form_result["matched_informative_fields"]
        informative_total += form_result["total_informative_fields"]

    score = matched_total / informative_total if informative_total else 0.0
    return {
        "score": score,
        "matched_informative_fields": matched_total,
        "total_informative_fields": informative_total,
        "hard_fail_reason": None,
        "per_form": per_form,
    }


def _load_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score taxform_2_1 outputs against reference payloads."
    )
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--forms", nargs="+", required=True)
    args = parser.parse_args()

    reference_payloads = {
        form_name: _load_payload(args.reference_dir / f"{form_name}_gt.json")
        for form_name in args.forms
    }
    output_payloads = {
        form_name: _load_payload(args.output_dir / f"{form_name}_output.json")
        for form_name in args.forms
    }
    print(json.dumps(score_variant_outputs(reference_payloads, output_payloads), indent=2))


if __name__ == "__main__":
    main()
