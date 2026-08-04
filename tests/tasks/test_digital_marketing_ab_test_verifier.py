from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

VERIFIER_PATH = (
    Path(__file__).resolve().parents[2]
    / "tasks/business_finance/digital_marketing_ab_test_analysis_1/scripts/verify_ab_test_outputs.py"
)
SPEC = importlib.util.spec_from_file_location("digital_marketing_ab_verifier", VERIFIER_PATH)
assert SPEC is not None and SPEC.loader is not None
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)
extract_recommendation = VERIFIER.extract_recommendation
validate_report_md = VERIFIER.validate_report_md


@pytest.mark.parametrize(
    ("section", "expected"),
    [
        ("Recommendation: SHIP", "ship"),
        ("## Recommendation\n**HOLD** the rollout.", "hold"),
        ("## Recommendation\nHold rather than ship it now.", "hold"),
        ("## Recommendation\nDo not ship this rollout.", "hold"),
        ("## Recommendation\nShip this rollout; do not hold it.", "ship"),
        ("## Recommendation\nShip or hold after another review.", None),
    ],
)
def test_extract_recommendation_resolves_section_semantics(
    section: str, expected: str | None
) -> None:
    assert extract_recommendation(section) == expected


def test_extract_recommendation_ignores_ship_outside_recommendation() -> None:
    report = "A ship carries cargo.\n\n## Recommendation\nHOLD the rollout."

    assert extract_recommendation(report) == "hold"


@pytest.mark.parametrize(
    ("recommendation", "expected"),
    [
        ("SHIP: launch the treatment.", True),
        ("HOLD: collect reliable click data first.", False),
        ("Hold the rollout rather than ship it now.", False),
        ("Do not ship this rollout.", False),
    ],
)
def test_validate_report_rejects_semantically_opposite_recommendations(
    tmp_path, recommendation: str, expected: bool
) -> None:
    report = tmp_path / "experiment_report.md"
    report.write_text(
        "Required per-arm sample size: 3,122\n\n"
        "Observed primary lift: 3.40 percentage points.\n\n"
        f"## Recommendation\n{recommendation}\n",
        encoding="utf-8",
    )
    expected_stats = {
        "opened_rate": {"significant_at_05": True},
        "unsubscribed_rate": {"absolute_lift": 0.002722},
    }

    assert validate_report_md(report, expected_stats, required_n=3122) is expected
