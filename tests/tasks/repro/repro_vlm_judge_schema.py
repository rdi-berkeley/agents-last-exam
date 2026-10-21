"""End-to-end repro: VLM judge payload schema enforcement.

Drives the REAL _call_vlm_judge with the judge transport replaced by a
local fake (the same seam pytest uses). Demonstrates the pre-fix behavior
for the historical record and asserts the fix: schema-drift payloads are
rejected as JudgeInfrastructureError instead of being averaged into a
score.

Run:  .venv/bin/python tests/tasks/repro/repro_vlm_judge_schema.py
Exit 0 when the fix holds (all hostile payloads rejected, valid payload
scores normally).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from tasks.engineering.robotics_blender_tabletop_reconstruction import main as task  # noqa: E402
from tasks.utils.evaluation import JudgeInfrastructureError  # noqa: E402

try:
    DIMS = task._VLM_DIMENSIONS
except AttributeError:  # pre-fix module has no dimension tuple
    DIMS = (
        "object_count",
        "spatial_layout",
        "camera_perspective",
        "material_fidelity",
        "lighting_quality",
        "table_surface",
    )


def full(**overrides):
    payload = {k: 1.0 for k in DIMS}
    payload.update(overrides)
    return payload


def main() -> int:
    cases = [
        ("valid payload", full(), "score", None),
        ("single wrong key", {"overall": 1.0}, "reject", None),
        ("missing keys", {"object_count": 1.0}, "reject", None),
        ("out-of-range 7.0", full(object_count=7.0), "reject", None),
        ("negative value", full(lighting_quality=-0.5), "reject", None),
        ("extra key alongside valid", full(overall=1.0), "reject", None),
        ("honest zeros", full(**{k: 0.0 for k in DIMS}), "score", 0.0),
    ]
    ok = True
    for name, payload, expect, expected_value in cases:
        with patch.object(task, "llm_multimodal_json_sync", return_value=payload):
            try:
                score = task._call_vlm_judge("ref", "cand")
                outcome = "scored"
                detail = f"{score:.3f}"
                passed = expect == "score" and (
                    expected_value is None or abs(score - expected_value) < 1e-9
                )
            except JudgeInfrastructureError as e:
                outcome = "rejected"
                detail = str(e)[:60]
                passed = expect == "reject"
        if not passed:
            ok = False
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {outcome} ({detail})")

    print(
        "RESULT:",
        "HARDENED - schema-drift payloads rejected, valid payloads score normally"
        if ok
        else "VULNERABLE - at least one hostile payload was averaged into a score",
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
