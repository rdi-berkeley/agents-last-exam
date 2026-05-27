#!/usr/bin/env python
"""Scorer for physical_sciences/adapt_vqe_molecular_energy."""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


TIERS = {
    "tier1": {
        "molecule": "H2",
        "threshold_ha": 1.0e-4,
        "allowed_methods": {"VQE", "ADAPT-VQE"},
        "require_adapt": False,
    },
    "tier2": {
        "molecule": "LiH",
        "threshold_ha": 1.6e-3,
        "allowed_methods": {"ADAPT-VQE"},
        "require_adapt": True,
    },
    "tier3": {
        "molecule": "BeH2",
        "threshold_ha": 1.6e-3,
        "allowed_methods": {"ADAPT-VQE"},
        "require_adapt": True,
    },
}

OPERATOR_RE = re.compile(r"^[SD]\([^)]+\)$")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _score_tier(tier_name: str, output_payload: Dict[str, Any], reference_payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
    spec = TIERS[tier_name]
    reasons: List[str] = []
    payload = output_payload.get(tier_name)
    if not isinstance(payload, dict):
        return False, [f"{tier_name}: missing object payload"]

    if payload.get("molecule") != spec["molecule"]:
        reasons.append(f"{tier_name}: molecule={payload.get('molecule')!r} expected {spec['molecule']!r}")

    energy = payload.get("energy_ha")
    target = reference_payload[tier_name]["exact_energy_ha"]
    if not _is_number(energy):
        reasons.append(f"{tier_name}: energy_ha not numeric ({energy!r})")
    elif abs(float(energy) - float(target)) > spec["threshold_ha"]:
        reasons.append(
            f"{tier_name}: energy error {abs(float(energy) - float(target)):.6e} "
            f"> {spec['threshold_ha']:.6e}"
        )

    method = payload.get("method")
    if method not in spec["allowed_methods"]:
        reasons.append(f"{tier_name}: method={method!r} not in {sorted(spec['allowed_methods'])}")

    n_parameters = payload.get("n_parameters")
    if not _is_positive_int(n_parameters):
        reasons.append(f"{tier_name}: n_parameters={n_parameters!r} must be positive int")

    if spec["require_adapt"]:
        adapt_iterations = payload.get("adapt_iterations")
        if not _is_positive_int(adapt_iterations):
            reasons.append(f"{tier_name}: adapt_iterations={adapt_iterations!r} must be positive int")

        operator_sequence = payload.get("operator_sequence")
        if not isinstance(operator_sequence, list) or not operator_sequence:
            reasons.append(f"{tier_name}: operator_sequence must be a non-empty list")
        else:
            for idx, operator in enumerate(operator_sequence):
                if not isinstance(operator, str) or not OPERATOR_RE.match(operator):
                    reasons.append(f"{tier_name}: operator_sequence[{idx}] invalid ({operator!r})")
                    break
        if (
            isinstance(adapt_iterations, int)
            and isinstance(n_parameters, int)
            and isinstance(operator_sequence, list)
            and len(operator_sequence) != adapt_iterations
        ):
            reasons.append(
                f"{tier_name}: len(operator_sequence)={len(operator_sequence)} "
                f"!= adapt_iterations={adapt_iterations}"
            )
        if (
            isinstance(adapt_iterations, int)
            and isinstance(n_parameters, int)
            and adapt_iterations != n_parameters
        ):
            reasons.append(
                f"{tier_name}: adapt_iterations={adapt_iterations} != n_parameters={n_parameters}"
            )

    return len(reasons) == 0, reasons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-file", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    reference_file = Path(args.reference_file)
    output_file = output_dir / "results.json"

    if not output_file.exists():
        print(json.dumps({"score": 0.0, "passed": {}, "reasons": [f"missing {output_file}"]}))
        return 0
    if not reference_file.exists():
        print(json.dumps({"score": 0.0, "passed": {}, "reasons": [f"missing {reference_file}"]}))
        return 0

    try:
        output_payload = _load_json(output_file)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"score": 0.0, "passed": {}, "reasons": [f"unparseable results.json: {exc}"]}))
        return 0

    try:
        reference_payload = _load_json(reference_file)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"score": 0.0, "passed": {}, "reasons": [f"unparseable reference results.json: {exc}"]}))
        return 0

    if not isinstance(output_payload, dict):
        print(json.dumps({"score": 0.0, "passed": {}, "reasons": ["results.json is not an object"]}))
        return 0

    passed: Dict[str, bool] = {}
    reasons: List[str] = []
    for tier_name in ("tier1", "tier2", "tier3"):
        ok, tier_reasons = _score_tier(tier_name, output_payload, reference_payload)
        passed[tier_name] = ok
        reasons.extend(tier_reasons)

    score = sum(1 for ok in passed.values() if ok) / 3.0
    print(json.dumps({"score": score, "passed": passed, "reasons": reasons}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
