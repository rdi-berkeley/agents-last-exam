from __future__ import annotations

import importlib.util
from pathlib import Path

VERIFIER_PATH = (
    Path(__file__).resolve().parents[2]
    / "tasks/psychology_neuro/scene2_resample/scripts/verify_outputs.py"
)
SPEC = importlib.util.spec_from_file_location("scene2_resample_verifier", VERIFIER_PATH)
assert SPEC is not None and SPEC.loader is not None
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)
read_stats = VERIFIER.read_stats
score_stats = VERIFIER.score_stats


def test_read_stats_ignores_descriptive_columns(tmp_path: Path) -> None:
    stats = tmp_path / "scene2_stats.csv"
    stats.write_text(
        "voxel_count,mean,max,median,stddev,Segment\n"
        "6906,-0.863581,3.261604,-0.771496,1.009246,roi_mask\n",
        encoding="utf-8",
    )

    assert read_stats(stats) == {
        "voxel_count": 6906.0,
        "mean": -0.863581,
        "max": 3.261604,
    }


def test_unreadable_csv_preserves_derived_mask_credit() -> None:
    expected = {"voxel_count": 6906.0, "mean": -0.863581, "max": 3.261604}

    payload = score_stats(expected, expected, None)

    assert payload == {
        "score": 0.7,
        "passed": False,
        "reasons": ["unreadable_stats_csv"],
    }
