"""Compute the required OpenFAST step-response summary metrics from `.outb` output."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
try:
    from openfast_io.FAST_output_reader import FASTOutputFile
except ModuleNotFoundError:  # pragma: no cover - user-facing dependency guard
    FASTOutputFile = None


SUMMARY_COLUMNS = [
    "max_gen_speed_rpm_60_120s",
    "settling_time_s_after_step",
    "mean_gen_pwr_kw_150_180s",
    "max_abs_tower_top_fa_m_0_180s",
    "mean_collective_pitch_deg_150_180s",
]


def _select_column(df: pd.DataFrame, candidates: list[str]) -> str:
    normalized = {col.split("_[", 1)[0]: col for col in df.columns}
    for name in candidates:
        if name in normalized:
            return normalized[name]
    raise KeyError(f"missing required columns; looked for {candidates}, available={list(df.columns)}")


def compute_summary(outb_path: Path) -> dict[str, float]:
    if FASTOutputFile is None:
        raise RuntimeError(
            "Missing dependency `openfast_io`. Run this script with "
            "`uv run --with openfast-io python ...`."
        )
    df = FASTOutputFile(str(outb_path), method="pandas").toDataFrame()

    time_col = _select_column(df, ["Time"])
    gen_speed_col = _select_column(df, ["GenSpeed"])
    gen_pwr_col = _select_column(df, ["GenPwr"])
    tower_fa_col = _select_column(df, ["TTDspFA"])
    pitch1_col = _select_column(df, ["BldPitch1"])
    pitch2_col = _select_column(df, ["BldPitch2"])
    pitch3_col = _select_column(df, ["BldPitch3"])

    time = df[time_col]
    gen_speed = df[gen_speed_col]
    tower_fa = df[tower_fa_col].abs()
    collective_pitch = (df[pitch1_col] + df[pitch2_col] + df[pitch3_col]) / 3.0

    win_60_120 = df[(time >= 60.0) & (time <= 120.0)]
    win_150_180 = df[(time >= 150.0) & (time <= 180.0)]
    win_0_180 = df[(time >= 0.0) & (time <= 180.0)]

    if win_60_120.empty or win_150_180.empty or win_0_180.empty:
        raise ValueError("required evaluation windows are missing from the OpenFAST output")

    final_mean_speed = win_150_180[gen_speed_col].mean()
    tolerance = abs(final_mean_speed) * 0.005
    tail = df[time >= 60.0].copy()
    within = (tail[gen_speed_col] - final_mean_speed).abs() <= tolerance
    suffix_all_true = within.iloc[::-1].cummin().iloc[::-1]
    settling_candidates = tail.loc[suffix_all_true, time_col]
    if settling_candidates.empty:
        settling_time = float(time.iloc[-1])
    else:
        settling_time = float(settling_candidates.iloc[0])

    summary = {
        "max_gen_speed_rpm_60_120s": float(win_60_120[gen_speed_col].max()),
        "settling_time_s_after_step": settling_time,
        "mean_gen_pwr_kw_150_180s": float(win_150_180[gen_pwr_col].mean()),
        "max_abs_tower_top_fa_m_0_180s": float(win_0_180[tower_fa_col].abs().max()),
        "mean_collective_pitch_deg_150_180s": float(
            (
                win_150_180[pitch1_col]
                + win_150_180[pitch2_col]
                + win_150_180[pitch3_col]
            ).mean()
            / 3.0
        ),
    }
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outb", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = compute_summary(args.outb)

    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
            writer.writeheader()
            writer.writerow(summary)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
