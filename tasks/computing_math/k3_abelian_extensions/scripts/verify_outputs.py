#!/usr/bin/env python
"""Verifier for computing_math/k3_abelian_extensions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_TOP_LEVEL_KEYS = {
    "total_extensions",
    "extensions",
    "non_product_type_count",
    "non_product_type",
}
EXPECTED_EXTENSION_KEYS = {
    "m",
    "G_invariant_factors",
    "G_order",
    "product_type",
}


class VerificationError(ValueError):
    """Raised when the submission payload violates the task contract."""


def _product(values: list[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def _ensure_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise VerificationError(f"{field_name} must be an integer")
    return value


def _canonicalize_invariant_factors(raw: Any, *, field_name: str) -> list[int]:
    if not isinstance(raw, list) or not raw:
        raise VerificationError(f"{field_name} must be a non-empty list")
    factors = []
    for index, item in enumerate(raw):
        value = _ensure_int(item, field_name=f"{field_name}[{index}]")
        if value <= 0:
            raise VerificationError(f"{field_name}[{index}] must be positive")
        factors.append(value)
    for left, right in zip(factors, factors[1:]):
        if left > right or right % left != 0:
            raise VerificationError(
                f"{field_name} must be in ascending invariant-factor order with divisibility"
            )
    return factors


def canonicalize_extension(entry: Any, *, list_name: str, index: int) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise VerificationError(f"{list_name}[{index}] must be a JSON object")
    keys = set(entry.keys())
    if keys != EXPECTED_EXTENSION_KEYS:
        raise VerificationError(
            f"{list_name}[{index}] must have exactly keys {sorted(EXPECTED_EXTENSION_KEYS)}"
        )

    m = _ensure_int(entry["m"], field_name=f"{list_name}[{index}].m")
    if m <= 0:
        raise VerificationError(f"{list_name}[{index}].m must be positive")

    factors = _canonicalize_invariant_factors(
        entry["G_invariant_factors"],
        field_name=f"{list_name}[{index}].G_invariant_factors",
    )

    g_order = _ensure_int(entry["G_order"], field_name=f"{list_name}[{index}].G_order")
    if g_order <= 0:
        raise VerificationError(f"{list_name}[{index}].G_order must be positive")
    if _product(factors) != g_order:
        raise VerificationError(
            f"{list_name}[{index}].G_order must equal the product of G_invariant_factors"
        )

    product_type = entry["product_type"]
    if not isinstance(product_type, bool):
        raise VerificationError(f"{list_name}[{index}].product_type must be boolean")

    return {
        "m": m,
        "G_invariant_factors": factors,
        "G_order": g_order,
        "product_type": product_type,
    }


def _extension_sort_key(entry: dict[str, Any]) -> tuple[int, tuple[int, ...]]:
    return int(entry["m"]), tuple(int(x) for x in entry["G_invariant_factors"])


def canonicalize_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise VerificationError("top-level payload must be a JSON object")
    keys = set(payload.keys())
    if keys != EXPECTED_TOP_LEVEL_KEYS:
        raise VerificationError(
            f"payload must have exactly keys {sorted(EXPECTED_TOP_LEVEL_KEYS)}"
        )

    total_extensions = _ensure_int(payload["total_extensions"], field_name="total_extensions")
    non_product_type_count = _ensure_int(
        payload["non_product_type_count"],
        field_name="non_product_type_count",
    )
    if total_extensions < 0 or non_product_type_count < 0:
        raise VerificationError("top-level counts must be non-negative")

    if not isinstance(payload["extensions"], list):
        raise VerificationError("extensions must be a list")
    if not isinstance(payload["non_product_type"], list):
        raise VerificationError("non_product_type must be a list")

    extensions = [
        canonicalize_extension(entry, list_name="extensions", index=index)
        for index, entry in enumerate(payload["extensions"])
    ]
    non_product_type = [
        canonicalize_extension(entry, list_name="non_product_type", index=index)
        for index, entry in enumerate(payload["non_product_type"])
    ]

    extensions = sorted(extensions, key=_extension_sort_key)
    non_product_type = sorted(non_product_type, key=_extension_sort_key)

    extension_keys = [(_extension_sort_key(entry), entry) for entry in extensions]
    if len({key for key, _ in extension_keys}) != len(extension_keys):
        raise VerificationError("extensions must not contain duplicate (m, G_invariant_factors) pairs")

    non_product_keys = [(_extension_sort_key(entry), entry) for entry in non_product_type]
    if len({key for key, _ in non_product_keys}) != len(non_product_keys):
        raise VerificationError(
            "non_product_type must not contain duplicate (m, G_invariant_factors) pairs"
        )

    if total_extensions != len(extensions):
        raise VerificationError("total_extensions must equal len(extensions)")
    if non_product_type_count != len(non_product_type):
        raise VerificationError("non_product_type_count must equal len(non_product_type)")

    extension_lookup = {key: entry for key, entry in extension_keys}
    for key, entry in non_product_keys:
        if key not in extension_lookup:
            raise VerificationError("every non_product_type entry must also appear in extensions")
        if extension_lookup[key] != entry:
            raise VerificationError(
                "non_product_type entries must match the corresponding extensions exactly"
            )
        if entry["product_type"]:
            raise VerificationError("non_product_type entries must all have product_type=false")

    derived_non_product = sorted(
        [entry for entry in extensions if not entry["product_type"]],
        key=_extension_sort_key,
    )
    if non_product_type != derived_non_product:
        raise VerificationError(
            "non_product_type must equal the filtered subset of extensions with product_type=false"
        )

    return {
        "total_extensions": total_extensions,
        "extensions": extensions,
        "non_product_type_count": non_product_type_count,
        "non_product_type": non_product_type,
    }


def verify_submission_texts(agent_text: str, reference_text: str) -> dict[str, Any]:
    try:
        agent_payload = canonicalize_payload(json.loads(agent_text))
        reference_payload = canonicalize_payload(json.loads(reference_text))
    except json.JSONDecodeError as exc:
        return {
            "score": 0.0,
            "passed": False,
            "reason": f"invalid JSON: {exc}",
        }
    except VerificationError as exc:
        return {
            "score": 0.0,
            "passed": False,
            "reason": str(exc),
        }

    if agent_payload != reference_payload:
        return {
            "score": 0.0,
            "passed": False,
            "reason": "canonicalized payload does not match reference",
            "agent_total_extensions": agent_payload["total_extensions"],
            "reference_total_extensions": reference_payload["total_extensions"],
            "agent_non_product_type_count": agent_payload["non_product_type_count"],
            "reference_non_product_type_count": reference_payload["non_product_type_count"],
        }

    return {
        "score": 1.0,
        "passed": True,
        "reason": "exact match",
    }


def verify_submission_files(agent_file: Path, reference_file: Path) -> dict[str, Any]:
    return verify_submission_texts(
        agent_file.read_text(encoding="utf-8-sig"),
        reference_file.read_text(encoding="utf-8-sig"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-file", required=True, type=Path)
    parser.add_argument("--reference-file", required=True, type=Path)
    args = parser.parse_args()
    report = verify_submission_files(args.agent_file, args.reference_file)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
