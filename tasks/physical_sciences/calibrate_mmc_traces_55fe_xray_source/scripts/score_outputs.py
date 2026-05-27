#!/usr/bin/env python
"""Scoring logic for physical_sciences/calibrate_mmc_traces_55fe_xray_source."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class ScoreReport:
    score: float
    reason: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json_load_from_path(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")
    return payload


def _json_load_from_bytes(payload: bytes) -> dict[str, Any]:
    decoded = payload.decode("utf-8")
    obj = json.loads(decoded)
    if not isinstance(obj, dict):
        raise ValueError("payload must be a JSON object")
    return obj


def _require_number(value: Any) -> float:
    if value is None:
        raise ValueError("missing value")
    if isinstance(value, bool):
        raise ValueError("boolean is not an allowed numeric value")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not numeric") from exc

    if not math.isfinite(number):
        raise ValueError("value must be finite")
    return number


def score_result_payload(candidate_bytes: bytes, reference_bytes: bytes) -> ScoreReport:
    """Return a binary score for the calibrated MMC output payload."""

    try:
        candidate = _json_load_from_bytes(candidate_bytes)
    except Exception as exc:
        return ScoreReport(0.0, "Candidate output JSON could not be read", {"reason": str(exc)})

    try:
        reference = _json_load_from_bytes(reference_bytes)
    except Exception as exc:
        return ScoreReport(0.0, "Reference output JSON could not be read", {"reason": str(exc)})

    candidate_keys = set(candidate.keys())
    reference_keys = set(reference.keys())
    if candidate_keys != {"fwhm", "fwhm_err"}:
        return ScoreReport(
            0.0,
            "Candidate payload must contain exactly fwhm and fwhm_err",
            {"candidate_keys": sorted(candidate_keys)},
        )

    for key in ("fwhm", "fwhm_err"):
        if key not in candidate:
            return ScoreReport(0.0, f"Missing candidate key: {key}", {"candidate_keys": sorted(candidate.keys())})
        if key not in reference:
            return ScoreReport(0.0, f"Missing reference key: {key}", {"reference_keys": sorted(reference.keys())})

    if reference_keys != {"fwhm", "fwhm_err"}:
        return ScoreReport(
            0.0,
            "Reference payload is not in the required schema",
            {"reference_keys": sorted(reference_keys)},
        )

    try:
        candidate_fwhm = _require_number(candidate["fwhm"])
        candidate_fwhm_err = _require_number(candidate["fwhm_err"])
        ref_fwhm = _require_number(reference["fwhm"])
        ref_fwhm_err = _require_number(reference["fwhm_err"])
    except Exception as exc:
        return ScoreReport(0.0, "Invalid numeric value", {"reason": str(exc)})

    tol_fwhm = 0.01
    tol_err = 0.02

    fwhm_diff = abs(candidate_fwhm - ref_fwhm)
    fwhm_err_diff = abs(candidate_fwhm_err - ref_fwhm_err)

    if fwhm_diff <= tol_fwhm and fwhm_err_diff <= tol_err:
        return ScoreReport(
            1.0,
            "Candidate matches expected FWHM result within tolerance",
            {
                "candidate": {"fwhm": candidate_fwhm, "fwhm_err": candidate_fwhm_err},
                "reference": {"fwhm": ref_fwhm, "fwhm_err": ref_fwhm_err},
                "diff": {"fwhm": fwhm_diff, "fwhm_err": fwhm_err_diff},
                "tolerance": {"fwhm": tol_fwhm, "fwhm_err": tol_err},
            },
        )

    return ScoreReport(
        0.0,
        "Candidate FWHM is outside tolerance",
        {
            "candidate": {"fwhm": candidate_fwhm, "fwhm_err": candidate_fwhm_err},
            "reference": {"fwhm": ref_fwhm, "fwhm_err": ref_fwhm_err},
            "diff": {"fwhm": fwhm_diff, "fwhm_err": fwhm_err_diff},
            "tolerance": {"fwhm": tol_fwhm, "fwhm_err": tol_err},
        },
    )


def main() -> None:  # pragma: no cover - helper for local manual checks
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference", required=True)
    args = parser.parse_args()

    report = score_result_payload(Path(args.candidate).read_bytes(), Path(args.reference).read_bytes())
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
